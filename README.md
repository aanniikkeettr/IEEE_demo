# Setup Instructions

Follow these steps to create a virtual environment and install the required packages.

## Prerequisites

Ensure you have **Python 3** installed on your system.

## Installation Steps

1. **Open your terminal** or command prompt.

2. **Navigate to your project directory** (replace `path/to/your/project` with your actual folder path):
   ```bash
   cd path/to/your/project
   ```

3. **Create a virtual environment** named `venv`:
   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment**:
   * On **macOS and Linux**:
     ```bash
     source venv/bin/activate
     ```
   * On **Windows** (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```
   * On **Windows** (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

5. **Upgrade pip** (optional but recommended):
   ```bash
   pip install --upgrade pip
   ```

6. **Install the packages**:
   ```bash
   pip install iqm-benchmarks iqm-qubit-selector matplotlib
   ```

## Verifying Installation

To verify that the packages were installed successfully, you can list your installed dependencies:
```bash
pip list
```
