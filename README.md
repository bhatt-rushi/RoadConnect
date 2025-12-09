# Graph-Based Sediment and Routing Model

This is a Graph-Based Sediment and Routing Model developed as part of an undergraduate capstone research project at the University of Texas at Austin.

---

## 🚧 Project Status

This project is in active development. The core routing model is functional, but many pre-processing steps and feature enhancements are still in experimental phases or not ready to be made public yet. Please note, this repo is currently outdated and many commits / refactors behind as it is still undergoing rapid iteration.

### Current Capabilities
The main program can currently:
* **Build a Directed Acyclic Graph (DAG)** using NetworkX based on pre-processed input files (roads, flowpaths, drains, etc.).
* **Model Runoff & Sediment:** For a given list of rainfall events, the model calculates how much runoff and sediment is generated from **road surfaces** and tracks its path to drains and ponds.
* **Calculate Travel Cost:** The model accounts for the "volume-to-breakthrough" cost for flow traveling over non-road surfaces (flowpaths).

### Current Input Requirements
The model currently requires all data to be pre-processed and provided in the correct format, as specified in the configuration file. For example, road shapefiles **must** already be segmented and include attributes for `ELEVATION`, `AREA`, and `TYPE`. `ELEVATION` and `AREA` will be automated in the near future.

-----

### Task Checklist

#### ✅ Completed

  * **Core Model Engine:** Implemented the core DAG (NetworkX) model to route runoff and sediment from pre-defined road segments to drains/ponds.
  * **Cost-Based Flow:** Implemented volume-to-breakthrough cost for non-road flowpaths.

#### ⬜ Remaining Tasks

**Pre-Processing & Data Integration**

  * [ ] **Road Elevation Filter:** Develop and implement a filter to smoothen road elevation profiles to handle noise.
  * [ ] **Integrate Road Processing:** Integrate module for automated road surface extraction and segmentation into the main program.
  * [ ] **Integrate DEM Automation:** Integrate module for automated downloading and caching of DEM data.
  * [ ] **Add Field Data:** Populate the configuration with real road types and erosion rates from field studies.

**Model Feature Enhancements**

  * [ ] **Polygon-Based Runoff:** Implement runoff and sediment generation from non-road polygons (e.g., agricultural land).
  * [ ] **Infiltration Modeling:** Experiment with and integrate infiltration curves to dynamically adjust breakthrough costs (only on non-road surfaces).
  * [ ] **Pond Filling:** Add sediment bulk density data to model the filling of ponds between rainfall events.
  * [ ] **Scenario Modeling:** Implement "what-if" analysis tools (e.g., testing optimal locations for new ponds).

**Usability & Documentation**

  * [ ] **Visualization Tools:** Develop tools to visualize model outputs and make results interpretable.
  * [ ] **Documentation:** Prepare training materials, usage examples, and a demo dataset.
  * [x] **Install Instructions:** Provide instructions on creating a venv with necessary packages.

-----

## 💾 Installation & Usage

1.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the Virtual Environment:**
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install the Project:**
    ```bash
    pip install .
    ```

4.  **Run the Model:**
    ```bash
    roadconnect
    ```

-----

## Author & Acknowledgements

This program was written by **Rushi Bhatt**.

Special thanks to:

  * **Professor Carlos E. Ramos Scharrón** for his mentorship and the foundational field studies that supported this research.
  * **Protectores de Cuencas Inc.** for providing partial support for this project with a stipend.

-----

## 📝 License

This program is licensed under the **GNU General Public License v3.0**.
