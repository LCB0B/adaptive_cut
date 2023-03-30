import graph_tool.all as gt
import numpy as np

name = "copenhagen/fb_friends" 
names = ["copenhagen/fb_friends",'facebook_friends',"unicodelang",'board_directors/net2m_2011-08-01','board_directors/net1m_2011-08-01',
           'ugandan_village/friendship-1','ugandan_village/friendship-2','ugandan_village/friendship-3','ugandan_village/friendship-4', 
           "euroroad","foodweb_baywet","product_space/HS","product_space/SITC","malaria_genes/HVR_1","malaria_genes/HVR_5","malaria_genes/HVR_6",
           "malaria_genes/HVR_7","malaria_genes/HVR_8","malaria_genes/HVR_9","cintestinalis","new_zealand_collab",
           "urban_streets/ahmedabad","urban_streets/bologna","urban_streets/brasilia","urban_streets/cairo","urban_streets/paris",
           "urban_streets/venice","urban_streets/vienna","kegg_metabolic/aae","kegg_metabolic/afu","kegg_metabolic/ana","kegg_metabolic/ape",
           "kegg_metabolic/atc","kegg_metabolic/ath","kegg_metabolic/atu","kegg_metabolic/bas","interactome_stelzl","interactome_figeys",
           "wiki_science",'interactome_vidal','bible_nouns','foursquare/NYC_restaurant_checkin','foursquare/NYC_restaurant_tips',
           "bitcoin_alpha",'plant_pol_robertson',"bitcoin_trust",'word_adjacency/japanese','word_adjacency/spanish','word_adjacency/french',
           "facebook_organizations/S1","facebook_organizations/S2","facebook_organizations/M1","facebook_organizations/L1",
           "physics_collab/pierreAuger","eu_procurements_alt/AT_2008","eu_procurements_alt/CZ_2008","eu_procurements_alt/ES_2008","eu_procurements_alt/EE_2008","eu_procurements_alt/DK_2011","eu_procurements_alt/FI_2008","eu_procurements_alt/GR_2008","eu_procurements_alt/HU_2015",
           "eu_procurements_alt/IT_2008","eu_procurements_alt/LT_2008","eu_procurements_alt/LV_2008","eu_procurements_alt/PL_2008","eu_procurements_alt/PT_2008",
           "eu_procurements_alt/SE_2008","eu_procurements_alt/SK_2008","gnutella/04","software_dependencies/jung-c","software_dependencies/slucene","software_dependencies/org",
           "software_dependencies/scolt","software_dependencies/jmail","software_dependencies/sjbullet","software_dependencies/jung","software_dependencies/colt",
           "software_dependencies/sjung","genetic_multiplex/Candida","genetic_multiplex/Gallus","genetic_multiplex/HumanHerpes4",
           "genetic_multiplex/HumanHIV1","genetic_multiplex/Plasmodium","genetic_multiplex/Rattus",'arxiv_collab/hep-th-1999',
           "tree-of-life/331678","tree-of-life/333990","tree-of-life/335543","tree-of-life/339670","tree-of-life/338969","tree-of-life/338966",
           "tree-of-life/340177","tree-of-life/365044","tree-of-life/366602","tree-of-life/406817",'tree-of-life/452652','tree-of-life/469383',
           "us_agencies/alabama","us_agencies/indiana","us_agencies/florida"]

def save_network(name):
    g = gt.collection.ns[name]
    g.set_directed(False)
    u = gt.extract_largest_component(g,directed=None)
    edges = u.get_edges()+1
    np.savetxt(f"data/data/{name.replace('/','-')}.txt", edges.astype(int),fmt='%i', delimiter="-")
    print(name)
    return


for name in names:
    save_network(name)  


# pos = gt.sfdp_layout(g)
# gt.graph_draw(g, pos=pos, output="graph-draw-arf.pdf")
# pos = gt.sfdp_layout(u)
# gt.graph_draw(g, pos=pos, output="graph-draw-arf_u.pdf")

#####

# https://networks.skewed.de/net/facebook_friends
# https://networks.skewed.de/net/unicodelang
# https://networks.skewed.de/net/board_directors
# https://networks.skewed.de/net/ugandan_village
# https://networks.skewed.de/net/euroroad
# https://networks.skewed.de/net/foodweb_baywet
# https://networks.skewed.de/net/product_space
# https://networks.skewed.de/net/malaria_genes
# https://networks.skewed.de/net/cintestinalis
# https://networks.skewed.de/net/new_zealand_collab
# https://networks.skewed.de/net/urban_streets
# https://networks.skewed.de/net/kegg_metabolic
# https://networks.skewed.de/net/interactome_stelzl
# https://networks.skewed.de/net/interactome_figeys
# https://networks.skewed.de/net/wiki_science
# https://networks.skewed.de/net/interactome_vidal
# https://networks.skewed.de/net/bible_nouns
# https://networks.skewed.de/net/foursquare
# https://networks.skewed.de/net/bitcoin_alpha
# https://networks.skewed.de/net/plant_pol_robertson
# https://networks.skewed.de/net/bitcoin_trust
# https://networks.skewed.de/net/word_adjacency
# https://networks.skewed.de/net/facebook_organizations
# https://networks.skewed.de/net/physics_collab
# https://networks.skewed.de/net/eu_procurements_alt
# https://networks.skewed.de/net/gnutella
# https://networks.skewed.de/net/software_dependencies
# https://networks.skewed.de/net/genetic_multiplex
# https://networks.skewed.de/net/arxiv_collab
# https://networks.skewed.de/net/tree-of-life
# https://networks.skewed.de/net/us_agencies