"""Scalar kernels shared by the pure-Python and the numba engines.

Every kernel below is written so that the very same source runs as plain
Python (on ``memoryview`` buffers) and, when numba is installed, as compiled
code. To keep the two engines bit-identical the kernels obey a few rules:
scalar arithmetic, indexing and ``math`` functions only; no numpy calls; no
calls between kernels; every buffer is preallocated by the caller as an
``int64`` or ``float64`` array.
"""

from __future__ import annotations

import functools
import math

LN2 = math.log(2.0)


def tree_tables(
    left, right, n, parent, size, start, order, minleaf, logZ, logZc, p_stop, p_both, p_lcum
):
    """Fill per-node tables of a binary dendrogram with ``n`` leaves.

    ``Z(v)`` is the sum of ``2^-|s|`` over all cuts ``s`` of the subtree of
    ``v`` and ``Z_c(v)`` the same sum restricted to cuts in which at least one
    child of ``v`` stays whole:

        Z(leaf) = 1/2,  Z(v) = 1/2 + Z(l) Z(r),  Z_c(v) = Z(l)/2 + Z(r)/2 - 1/4.

    The proposal probabilities derived from them are

        p_stop(v) = 1 / (2 Z(v))                stop at v in an unconstrained cut
        p_both(v) = 1 / (4 Z_c(v))              down move keeps both children
        p_lcum(v) = p_both + (Z(l) - 1/2) / (2 Z_c(v))   cumulative, splits l

    ``start``/``order`` give the left-to-right leaf order (equal to
    ``scipy.cluster.hierarchy.leaves_list``) so every subtree is the interval
    ``order[start[v]:start[v] + size[v]]``.
    """
    for v in range(n):
        size[v] = 1
        minleaf[v] = v
        logZ[v] = -LN2
        p_stop[v] = 1.0
    for i in range(n - 1):
        v = n + i
        l = left[i]
        r = right[i]
        parent[l] = v
        parent[r] = v
        size[v] = size[l] + size[r]
        minleaf[v] = min(minleaf[l], minleaf[r])
        a = -LN2
        b = logZ[l] + logZ[r]
        lz = max(a, b) + math.log1p(math.exp(-abs(a - b)))
        logZ[v] = lz
        a = logZ[l] - LN2
        b = logZ[r] - LN2
        ls = max(a, b) + math.log1p(math.exp(-abs(a - b)))
        lzc = ls + math.log1p(-0.25 * math.exp(-ls))
        logZc[v] = lzc
        p_stop[v] = math.exp(-LN2 - lz)
        if l < n and r < n:
            pb = 1.0
            pl = 1.0
        else:
            pb = math.exp(-2.0 * LN2 - lzc)
            if l < n:
                pl = pb
            elif r < n:
                pl = 1.0
            else:
                # Z(l) - 1/2 = Z(ll) Z(lr): no cancellation
                pl = pb + math.exp(logZ[left[l - n]] + logZ[right[l - n]] - LN2 - lzc)
        p_both[v] = pb
        p_lcum[v] = pl
    root = 2 * n - 2
    parent[root] = -1
    start[root] = 0
    for i in range(n - 2, -1, -1):
        v = n + i
        l = left[i]
        start[l] = start[v]
        start[right[i]] = start[v] + size[l]
    for v in range(n):
        order[start[v]] = v


def chain(
    left,
    right,
    parent,
    n,
    scores,
    logZc,
    p_stop,
    p_both,
    p_lcum,
    cut,
    pos,
    best,
    trace,
    stack,
    rem,
    add,
    u,
    ist,
    fst,
    T0,
    k_sa,
    n_steps,
    trace_every,
):
    """Run the reversible-jump Metropolis-Hastings chain over dendrogram cuts.

    State: ``cut[:k]`` holds the current cluster nodes and ``pos[v]`` the index
    of ``v`` in ``cut`` (or -1). ``ist = [step, k, upos, at_best, best_k,
    n_accepted]`` and ``fst = [f, best_f]`` carry the scalar state so the
    driver can resume after refilling the uniform buffer ``u``. The kernel
    returns at a step boundary once fewer than ``n + 4`` uniforms are left,
    which bounds the draws of any single step.

    Draw order per step: pick a cluster, [direction], [down-move case, one
    draw per internal node visited by the cut sampler], acceptance.
    """
    step = ist[0]
    k = ist[1]
    upos = ist[2]
    at_best = ist[3]
    best_k = ist[4]
    n_acc = ist[5]
    f = fst[0]
    best_f = fst[1]
    root = 2 * n - 2
    nu = len(u)
    while step < n_steps and upos + n + 4 <= nu:
        x = cut[min(int(u[upos] * k), k - 1)]
        upos += 1
        if x < n:
            up = True
        elif x == root:
            up = False
        else:
            up = u[upos] < 0.5
            upos += 1
        nr = 0
        na = 0
        lw = False
        rw = False
        if up:
            # merge x with its sibling s into the parent p; if s is split,
            # every cluster below s is collapsed as well
            p = parent[x]
            s = left[p - n] + right[p - n] - x
            pux = 1.0 if x < n else 0.5
            rem[0] = x
            nr = 1
            if pos[s] >= 0:
                rem[1] = s
                nr = 2
                pus = 1.0 if s < n else 0.5
                logqf = math.log(pux + pus) - math.log(k)
            else:
                ns = 1
                stack[0] = s
                while ns > 0:
                    ns -= 1
                    v = stack[ns]
                    if pos[v] >= 0:
                        rem[nr] = v
                        nr += 1
                    else:
                        stack[ns] = right[v - n]
                        stack[ns + 1] = left[v - n]
                        ns += 2
                logqf = math.log(pux) - math.log(k)
            k2 = k - nr + 1
            add[0] = p
            na = 1
            delta = scores[p]
            for j in range(nr):
                delta -= scores[rem[j]]
            pdp = 1.0 if p == root else 0.5
            logqr = math.log(pdp) - math.log(k2) - nr * LN2 - logZc[p]
        else:
            # split x into a cut sampled from 2^-|s| / Z_c(x): at least one
            # child of x stays whole
            l = left[x - n]
            r = right[x - n]
            uc = u[upos]
            upos += 1
            if uc < p_both[x]:
                lw = True
                rw = True
                add[0] = l
                add[1] = r
                na = 2
            else:
                if uc < p_lcum[x]:
                    c = l
                    rw = True
                else:
                    c = r
                    lw = True
                    add[0] = l
                    na = 1
                stack[0] = right[c - n]
                stack[1] = left[c - n]
                ns = 2
                while ns > 0:
                    ns -= 1
                    v = stack[ns]
                    if v < n:
                        add[na] = v
                        na += 1
                    else:
                        us = u[upos]
                        upos += 1
                        if us < p_stop[v]:
                            add[na] = v
                            na += 1
                        else:
                            stack[ns] = right[v - n]
                            stack[ns + 1] = left[v - n]
                            ns += 2
                if rw:
                    add[na] = r
                    na += 1
            rem[0] = x
            nr = 1
            k2 = k - 1 + na
            delta = -scores[x]
            for j in range(na):
                delta += scores[add[j]]
            pdx = 1.0 if x == root else 0.5
            logqf = math.log(pdx) - math.log(k) - na * LN2 - logZc[x]
            q = 0.0
            if lw:
                q += 1.0 if l < n else 0.5
            if rw:
                q += 1.0 if r < n else 0.5
            logqr = math.log(q) - math.log(k2)
        if step <= k_sa:
            T = T0
        else:
            T = T0 / math.log(step - k_sa + 2)
        la = delta / T + logqr - logqf
        ua = u[upos]
        upos += 1
        if la >= 0.0 or ua < math.exp(la):
            fnew = f + delta
            new_best = fnew > best_f
            if at_best == 1 and not new_best:
                # leaving the best state seen so far: snapshot it
                for j in range(k):
                    best[j] = cut[j]
                best_k = k
                at_best = 0
            for j in range(nr):
                v = rem[j]
                i = pos[v]
                last = cut[k - 1]
                cut[i] = last
                pos[last] = i
                pos[v] = -1
                k -= 1
            for j in range(na):
                v = add[j]
                cut[k] = v
                pos[v] = k
                k += 1
            f = fnew
            n_acc += 1
            if new_best:
                best_f = fnew
                at_best = 1
        if (step + 1) % trace_every == 0:
            trace[step // trace_every] = f
        step += 1
    ist[0] = step
    ist[1] = k
    ist[2] = upos
    ist[3] = at_best
    ist[4] = best_k
    ist[5] = n_acc
    fst[0] = f
    fst[1] = best_f


def union_find(n, a, b, h, uf, ufsize, cid, Z, row):
    """Kruskal merges over item pairs ``(a[t], b[t])`` given in merge order.

    Writes one linkage row ``[min id, max id, h[t], size]`` per successful
    union into ``Z`` starting at ``row`` and returns the next free row. The
    new cluster created by row ``i`` has id ``n + i`` (scipy's convention).
    """
    for t in range(len(a)):
        x = a[t]
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        y = b[t]
        while uf[y] != y:
            uf[y] = uf[uf[y]]
            y = uf[y]
        if x == y:
            continue
        if ufsize[x] < ufsize[y]:
            x, y = y, x
        cx = cid[x]
        cy = cid[y]
        uf[y] = x
        ufsize[x] += ufsize[y]
        Z[row, 0] = min(cx, cy)
        Z[row, 1] = max(cx, cy)
        Z[row, 2] = h[t]
        Z[row, 3] = ufsize[x]
        cid[x] = n + row
        row += 1
    return row


@functools.cache
def numba_available() -> bool:
    """True when numba imports cleanly (not merely installed)."""
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def resolve_engine(engine: str) -> str:
    if engine == "python":
        return "python"
    if engine == "auto":
        return "numba" if numba_available() else "python"
    if engine == "numba":
        if not numba_available():
            import numba  # noqa: F401  (re-raise the original import error)
        return "numba"
    raise ValueError(f"engine must be 'auto', 'numba' or 'python', got {engine!r}")


@functools.cache
def _jit(name: str, cache: bool = True):
    import numba

    return numba.njit(cache=cache, nogil=True)(globals()[name])


def call(name: str, engine: str, *args):
    """Run kernel ``name`` with the given engine.

    The Python engine receives ``memoryview`` buffers (zero-copy, in place)
    and plain Python scalars, so both engines do the same float arithmetic.
    """
    if engine == "numba":
        try:
            return _jit(name)(*args)
        except RuntimeError as err:  # e.g. read-only install: compile without caching
            if "cannot cache" not in str(err):
                raise
            return _jit(name, cache=False)(*args)
    conv = []
    for a in args:
        if hasattr(a, "__array_interface__"):
            conv.append(memoryview(a))
        elif isinstance(a, float):
            conv.append(float(a))
        else:
            conv.append(int(a))
    return globals()[name](*conv)
