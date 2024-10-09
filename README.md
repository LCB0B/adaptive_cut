# AdaptiveCut

## Preprint

 ```prepint.pdf ```
 
## Installation

To install the dependencies for this project, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone --depth 1 https://github.com/LCB0B/adaptive_cut.git
   cd adaptive_cut
   ```

2. **Create a virtual environment**:
   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. **Install the dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running Example Scripts

You can run the example scripts in the `examples` directory to see how the various methods work. For instance:
   ```sh
   python examples/toy_model_lc.py
   ```

### Plotting

To generate plots, you can run the scripts in the `plots` directory:
   ```sh
   python plots/plot_dendrogram.py
 ``` 

## Contributing

We welcome contributions to the project. To contribute, follow these steps:

1. **Fork the repository**:
   Click the "Fork" button on the top right of the repository page.

2. **Clone your fork**:
   ```sh
   git clone https://github.com/LCB0B/adaptive_cut.git
   cd adaptive_cut
   ```

3. **Create a new branch**:
   ```sh
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**:
   Implement your feature or fix.

5. **Commit your changes**:
   ```sh
   git add .
   git commit -m "Description of your changes"
   ```

6. **Push to your branch**:
   ```sh
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   Go to the original repository and create a pull request from your fork.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
