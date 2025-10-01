r"""°°°
# Building a sandbox environment
°°°"""
# |%%--%%| <tiMjrdetws|1D0cJTC9Mz>
r"""°°°
As part of my undergraduate capstone research project at UT Austin, I'm working on developing a model to calculate sediment delivery for road networks and watersheds in Western Puerto Rico and Culebra. To help aid in model development and testing, I aim to create a sandbox environment here where all model functionality can be tested.

Disclaimer: While my primary academic background is in Environmental Science, I hope to accurately discuss topics in other relevant domains as well.
°°°"""
# |%%--%%| <1D0cJTC9Mz|LZf0zQxUZb>
r"""°°°
## Learning about rasters
°°°"""
# |%%--%%| <LZf0zQxUZb|wRqtCrFf7G>
r"""°°°
It's simple enough to download data from the USGS and drag and drop it into your GIS software of choice, but, what if you wanted to create your own environment from scratch? Or perhaps you simply seek to better understand *what* a raster image is. Well, this will be a exercise in exactly that!

Let's start by importing some modules, hopefully you already have these installed or you know how to.
°°°"""
# |%%--%%| <wRqtCrFf7G|rNL61jfd3t>

import numpy as np # For dealing with arrays
import rasterio # For helping georef our mock data
from io import BytesIO # To have a place to write files to in-memory
import matplotlib.pyplot as plt # For plotting

# |%%--%%| <rNL61jfd3t|q5FFwNQ54P>
r"""°°°
### Generating some mock data
°°°"""
# |%%--%%| <q5FFwNQ54P|9eBAu9YrrN>
r"""°°°
A digital elevation model (DEM) is a raster data structure representing terrain elevation as a regular grid of uniformly sized cells, where each cell contains a z-value representing the surface elevation at its corresponding geographic (x,y) location. So, let's create a numpy array with such values:
°°°"""
# |%%--%%| <9eBAu9YrrN|che8xUXOhM>

dem = np.array([
    [10, 8, 6, 4, 2], 
    [20,16,12, 8, 4], 
    [40,32,24,16, 8], 
    [80,64,48,32,16]
    ])

plt.figure(figsize=(8, 6))
plt.imshow(dem, cmap='gray')
plt.colorbar(label='Pixel Intensity')
plt.title('Raster Image')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.xticks(range(dem.shape[1]))
plt.yticks(range(dem.shape[0]))
plt.tight_layout()
plt.show()

# |%%--%%| <che8xUXOhM|qSb4Zs1ITP>
r"""°°°
And just like that, we've created a raster image! However, lacking any geo-referencing, it's one that isn't helpful to a GIS specialist. Let's fix that by using a affine transform and rasterio!

We're going to use a standard 2D affine transform matrix to convert our pixel coordinates to real-world coordinates. Here's what that matrix looks like:

$$
\begin{bmatrix}
a & b & t_x \\
c & d & t_y \\
0 & 0 & 1
\end{bmatrix}
$$

Where:
- `a`: x-axis scaling
- `b`: x-axis rotation/shear
- `c`: y-axis rotation/shear
- `d`: y-axis scaling
- `t_x`: x-axis translation
- `t_y`: y-axis translation

WARNING! When it comes to representing this matrix as a 1-Dimensional array, different packages will sort the variables in a different order!

Transformation of point $(x,y)$ becomes:
$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix}
a & b & t_x \\
c & d & t_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

What's going on with the last row in the 3x3 matrix? Are we secretly working in 3D?? No, the reason for this is homogeneous coordinates. We represent the point (x,y) with an extra coordinate,

$$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

This allows us to simplify calculations as otherwise, we'd have to seperately perform operations to calculate scaling, rotation, and translation. Why?

Let $$A = \begin{bmatrix} a & b & t_x \\ d & e & t_y \end{bmatrix}$$ be a 2x3 matrix

Let $$B = \begin{bmatrix} x \\ y \end{bmatrix}$$ be a 2x1 matrix

Matrix multiplication requires the inner dimensions to match. Here, A is 2x3 and B is 2x1, so their inner dimensions (3 ≠ 2) do not align.

Therefore, $$A \times B$$ is not possible.

Let's run through an example. Let's define a transform:
°°°"""
# |%%--%%| <qSb4Zs1ITP|C1A3THMtVn>

transform = (
    0.50,        # a: x-axis pixel width (horizontal resolution)
    0.00,        # b: x-axis rotation/skew (typically zero for north-up images)
    320084.00,   # t_x: x-axis translation (easting origin)
    0.00,        # c: y-axis rotation/skew (typically zero for north-up images)
    -0.50,       # d: y-axis pixel height (vertical resolution, negative for top-down rasters)
    251741.00    # t_y: y-axis translation (northing origin)
)

# |%%--%%| <C1A3THMtVn|xmgoweGIBP>
r"""°°°
### Manually calculating the transform matrix for the pixel at (1,1)

X-coordinate: $x' = (a \cdot x) + (b \cdot y) + t_x = (0.50 \cdot 1) + (0.00 \cdot 1) + 320,084.00 = 320,084.50$

Y-coordinate: $y' = (c \cdot x) + (d \cdot y) + t_y = (0.00 \cdot 1) + (-0.50 \cdot 1) + 251,741.00 = 251,740.50$

Z-coordinate: $z' = z$ (elevation remains unchanged in this 2D affine transformation)

**Result**: Positions the **top-left** of the $(1,1)$ pixel at $(320,084.50, 251,740.50)$

Now, we'll use rasterio to automatically generate a georeferenced TIFF file from our raster data and transform, avoiding manual coordinate calculations.
°°°"""
# |%%--%%| <xmgoweGIBP|Lb8DRO7mro>

profile = {
    'driver': 'GTiff',
    'height': dem.shape[0],
    'width': dem.shape[1],
    'count': 1, # Defines the number of bands in the output raster
    'dtype': dem.dtype,
    'crs': '',
    'transform': transform,
}

# |%%--%%| <Lb8DRO7mro|vX5maXgL0r>
r"""°°°
One thing we haven't discussed yet is our coordinate reference system (CRS). Without a defined CRS, our transformation values are essentially meaningless. The transform's primary purpose is to convert coordinates between different spatial reference systems. In our specific example, we're translating from a pixel-based Cartesian coordinate system (which has no inherent geographic meaning) to a projected coordinate system, specifically EPSG 6566, which also uses a Cartesian coordinate system.

The EPSG system represents a standardized way of mapping coordinates to a specific region of the Earth's surface. EPSG 6566's base geographic coordinate system is NAD83(2011) (using the GRS 1980 ellipsoid), which uses ellipsoidal coordinates (lat/long). This is then projected to a flat surface using the Lambert conic conformal projection. Now all that's left is to choose what kind of grid to draw on this flat surface and where to have its origin. This is also the crucial part where we get to define our units of measurement (well, by 'we' I really mean the International Association of Oil & Gas Producers' Geomatics Committee who have already done all this work for us). For EPSG 6566, the coordinate system's origin is positioned southwest (ensuring positive values) off the coast of Puerto Rico, with measurements defined in meters.
<pre>
ESRI Well Known Text

PROJCS["NAD_1983_2011_StatePlane_Puerto_Rico_Virgin_Isls_FIPS_5200",
    GEOGCS["GCS_NAD_1983_2011",
        DATUM["D_NAD_1983_2011",
            SPHEROID["GRS_1980",6378137.0,298.257222101]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Lambert_Conformal_Conic"],
    PARAMETER["False_Easting",200000.0],
    PARAMETER["False_Northing",200000.0],
    PARAMETER["Central_Meridian",-66.4333333333333],
    PARAMETER["Standard_Parallel_1",18.4333333333333],
    PARAMETER["Standard_Parallel_2",18.0333333333333],
    PARAMETER["Latitude_Of_Origin",17.8333333333333],
    UNIT["Meter",1.0]]
</pre>
All that's left for us to do is add the EPSG code our profile defintion for rasterio! 
°°°"""
# |%%--%%| <vX5maXgL0r|skWk9DYRrZ>

profile = {
    'driver': 'GTiff',
    'height': dem.shape[0],
    'width': dem.shape[1],
    'count': 1, # Defines the number of bands in the output raster
    'dtype': dem.dtype,
    'crs': 'EPSG:6566',
    'transform': transform,
}

# Use rasterio to generate out tiff file in-memory and then read it back in
with BytesIO() as mem_file:
    with rasterio.open(mem_file, 'w', **profile) as dst:
        dst.write(dem, 1)

    mem_file.seek(0)

    with rasterio.open(mem_file) as src:
        raster_array = src.read(1)
        # Note how we can get geospatial metadata from a file
        transform = src.transform
        crs = src.crs

# Plot the raster
plt.figure(figsize=(10, 6))
plt.imshow(raster_array, cmap='gray',
# It'd be smarter to use src.bounds to get the extent but I want to show the math
           extent=[
               transform[2],  # x_min (left)
               transform[2] + src.shape[1] * transform[0],  # x_max (right)
               transform[5] + src.shape[0] * transform[4],  # y_min (bottom)
               transform[5]  # y_max (top)
           ])
plt.colorbar(label='Elevation')
plt.title(f'Geo-Referenced Raster\nCRS: {crs}')
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.tight_layout()
plt.show()

print(f"Affine Transform:\n{transform}")
print(f"\nCoordinate Reference System: {crs}")

# |%%--%%| <skWk9DYRrZ|sqLDTGRXbW>
r"""°°°
We covered a very basic transform in this section, the 2D affine transformation. I presented the concepts as they were relevant to GIS, but the affine transformation is widely used in computer graphics to manipulate objects in 2D or 3D space (uses a 4x4 matrix). This is because pixel coordinates are discrete and linear and screens are flat 2D surfaces (yes, I am aware of curved ultrawides). In order to convert coordinates between different EPSG standards, we'd require much more complex transformations as they are non-linear transformations accounting for different earth models, datum shifts, and projection methods. These transformations usually suffer a loss in accuracy due to the different representation of space requiring values to be resampled. Keep this in mind starting a new GIS project and having data in different CRSs!
°°°"""
# |%%--%%| <sqLDTGRXbW|bpBNGpESZj>

# Cleaning up variables from this section as they will not be used moving forward

%reset -f

# |%%--%%| <bpBNGpESZj|teeNX3sWZL>
r"""°°°
## Starting Model Design
°°°"""
#|%%--%%| <teeNX3sWZL|G1nXPSsrbX>
r"""°°°
Let's create a class to represent our model. This section of code will serve as a preview to everything that we'll be building.

My model design really isn't something I'd recommend trying to learn from. I'm not entirely sure that I've chosen the best design for the class structure and I've spent days arguing with myself on how things should be organized and stored in the model. If there is anything for you to learn here, it should be this: just code. If you find out that things need to change in the future, it's only because you made it that far in the first place to have discovered those problems. Think a little, of course, but type more. Speaking of... I need a new keyboard....

You'll also notice some extra helper functions. These functions will help us to plot/load/save raster data. Also, there's a neat little function to help us visualize the raster in 3D.
°°°"""
# |%%--%%| <G1nXPSsrbX|gGD82UYvMp>

try:
    import os
    from io import BytesIO
    from typing import Optional, Union, NamedTuple, List

    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import open3d as o3d
    import pyproj
    import rasterio
    import rasterstats
    import richdem as rd
    import scipy
    import seaborn as sns
    import shapely
    import skimage

except ImportError:
    print("Required libraries not installed!")
    raise


class Model:
    def __init__(
        self,
        crs: Optional[int] = None,
        elevation_path: Optional[str] = None,
        roads_path: Optional[str] = None,
        **kwargs
    ):

        # crs
        self.crs: pyproj.CRS = None
        if isinstance(crs, int):
            self.crs = pyproj.CRS.from_user_input(crs)
        elif crs is None:  # Pass so we can read in the CRS from the elevation file or assume that we'll be generating data
            pass
        else:
            raise TypeError(f"CRS must be an integer or None. Received: {type(crs)} with value {crs}")

        # elevation
        self.elevation: Elevation = Elevation(self)
        if isinstance(elevation_path, str):
            self.elevation.load(elevation_path)

        # roads
        self.roads: Road = Road(self)
        if isinstance(roads_path, str):
            self.roads.load(roads_path)


class Elevation(Model):
    def __init__(
        self,
        model=None,
        array=None,
        profile=None,
        transform=None,
        bounds=None,
        path=None,
    ):
        self.model = model
        self.array = array
        self.profile = profile
        self.transform = transform
        self.bounds = bounds

        if path is not None:
            self.load(path)

    def load(
        self,
        path: str,
    ):
        r"""
        Loads a Digital Elevation Model (DEM) with rasterio. Check rasterio for more detailed documentation.

        Parameters:
        -----------
        path : string
            Path to the raster file
        """
        # TODO: Reproject if self.model.crs already has a CRS setup
        try:
            with rasterio.open(path) as src:
                self.array = src.read(1)
                self.profile = src.profile
                self.model.crs = src.crs
                self.transform = src.transform
                self.bounds = src.bounds
                self.nodata = src.nodata

            return self
        except Exception as e:
            raise RuntimeError(f"Couldn't load the DEM raster from {path}: {e}")

    def save(
        self,
        path: str
    ):
        r"""
        Saves a Digital Elevation Model (DEM) with rasterio.

        Parameters:
        -----------
        path : string
            Path to the output raster file
        """
        try:
            # Ensure we have the necessary attributes to save
            if not hasattr(self, 'array') or not hasattr(self, 'profile'):
                raise AttributeError("No elevation data available to save")

            # Write the raster
            with rasterio.open(path, 'w', **self.profile) as dst:
                dst.write(self.array, 1)

        except Exception as e:
            raise RuntimeError(f"Couldn't save the DEM raster to {path}: {e}")

        return self

    def plot(
        self,
        cmap='gray',
        figsize=(10, 6),
        title=None,
        show_colorbar=True
    ):
        """
        Plot the elevation with customizable parameters.

        Parameters:
        -----------
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use for the plot (default is 'gray')
        figsize : tuple, optional
            Figure size (width, height) in inches (default is (10, 6))
        title : str, optional
            Custom title for the plot. If None, uses default title with CRS
        show_colorbar : bool, optional
            Whether to display the colorbar (default is True)

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """

        plt.figure(figsize=figsize)

        im = plt.imshow(self.array, cmap=cmap, extent=self.bounds)

        if title is None:
            title = f'Elevation\nCRS: {self.model.crs.to_epsg()}'
        plt.title(title)

        if show_colorbar:
            plt.colorbar(im, label='Elevation')

        plt.xlabel('Easting')
        plt.ylabel('Northing')
        plt.tight_layout()
        plt.show()

        return self

    def plot3d(
        self,
        cmap='terrain', 
        dark_background=True,
        add_coordinate_frame=True,
    ):
        """
        Visualize the elevation data as a 3D mesh using Open3D with customizable rendering options.

        Parameters:
        -----------
        palette : str, optional
            Seaborn color palette to use for coloring the mesh based on elevation.
            Default is 'terrain', which provides a natural-looking elevation color gradient.
        dark_background : bool, optional
            If True, sets the visualization background to black.
            Default is True for enhanced contrast and aesthetic appeal.
        add_coordinate_frame : bool, optional
            If True, displays the coordinate frame axes in the visualization.
            Default is True to provide spatial orientation.

        Returns:
        --------
        None
            Opens an interactive 3D visualization window that can be manually closed.

        Notes:
        ------
        - The method creates a triangulated mesh from the elevation array
        - Vertex colors are generated based on the height of each point
        - The color palette maps elevation values to a gradient of colors

        Examples:
        ---------
        >>> m.elevation.plot3d()  # Default visualization
        >>> m.elevation.plot3d(palette='viridis', dark_background=False)
        >>> m.elevation.generate(width=500,height=500,slope=0.20).plot3d()
        """
        os.environ['XDG_SESSION_TYPE'] = 'x11'

        height, width = self.array.shape

        # Create vertices
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        vertices = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            self.array.ravel()
        ])

        # Create triangulation
        triangles = []
        for i in range(height - 1):
            for j in range(width - 1):
                # First triangle
                triangles.append([
                    i * width + j, 
                    i * width + j + 1, 
                    (i + 1) * width + j
                ])
                # Second triangle
                triangles.append([
                    i * width + j + 1, 
                    (i + 1) * width + j + 1, 
                    (i + 1) * width + j
                ])

        # Convert to Open3D mesh
        vertices = o3d.utility.Vector3dVector(vertices)
        triangles = o3d.utility.Vector3iVector(triangles)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = vertices
        mesh.triangles = triangles

        # Compute vertex colors using Seaborn palette
        height_min = np.min(self.array)
        height_max = np.max(self.array)

        # Generate Seaborn color palette
        n_colors = 256  # High resolution for smooth gradient
        seaborn_palette = sns.color_palette(cmap, n_colors)

        vertex_colors = [seaborn_palette[np.clip(int((v[2] - height_min) / (height_max - height_min) * (len(seaborn_palette) - 1)), 0, len(seaborn_palette) - 1)] for v in mesh.vertices]

        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        mesh.compute_vertex_normals()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)

        render_option = vis.get_render_option()

        if dark_background:
            render_option.background_color = np.asarray([0, 0, 0])

        if add_coordinate_frame:
            render_option.show_coordinate_frame = True

        # Update and run visualization
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()

    def generate_valley_elevation(
        self,
        width:int=100, height:int=100,
        flank_slope:float=0.1,
        base_region_elevation:Union[int,float]=0, base_region_percent:float=0.1,
        crs:int=6566, origin_x:float=320084.0, origin_y:float=251741.0,
        pixel_width:float=0.5, pixel_height:float=-0.5,
    ):
        r"""
        Create a Digital Elevation Model (DEM) with a terrain profile that looks like \_/.

        Parameters:
        -----------
        width : int
            Width of the DEM raster (default: 100)
        height : int
            Height of the DEM raster (default: 100)
        slope : float
            Slope gradient for terrain profile (default: 0.1)
        base_region_elevation : int or float
            Elevation of center regions (default: 0)
        base_region_percent : float
            Percent of raster width used by underscore region (default: 0.1)
        crs : int
            EPSG Code as in integer. Note, if you update this, you also have to update origin_x/y for meaningful results. (default: 6566)
        origin_x : float
            X-coordinate of the raster origin (default: 320084.0)
        origin_y : float
            Y-coordinate of the raster origin (default: 251741.0)
        pixel_width : float
            Horizontal pixel resolution (default: 0.5)
        pixel_height : float
            Vertical pixel resolution (negative for top-down rasters) (default: -0.5)
        """
        raise NotImplementedError(
            "Method must be implemented in a subsequent cell."
        )

    def generate_local_features(
        self,
        exclusion_mask: np.ndarray | None = None,
        gauss_norm_distribution_mu=0, gauss_norm_distribution_sigma=0.5, 
        gauss_smoothness_filter_sigma=3, 
        should_apply_gaussian_smoothing = True, smooth_exclusion_regions = False,
        grid_shape=(10,10)):
            raise NotImplementedError(
                "Method must be implemented in a subsequent cell."
            )

    def identify_local_min_max(
        self,
        footprint: np.ndarray = scipy.ndimage.generate_binary_structure(2, 2)
    ):
        r"""

        Parameters:
        -----------
        footprint : np.ndarry
            Footprint over which to apply min/max filter (default: 3x3)
        """
        raise NotImplementedError(
            "Method must be implemented in a subsequent cell."
        )

class RoadParametersStruct(NamedTuple):
    name: str
    runoff_coeff: float
    erosion_coeff: float

class Road(Model):
    def __init__(
        self,
        model: Model,
        types: List[RoadParametersStruct] | None = None,
        transform: skimage.transform.AffineTransform | None = None,
        path: str | None = None,
    ):
        self.model = model
        self.types = types
        self.lines_gdf = gpd.GeoDataFrame(columns=['geometry', 'elevation', 'width', 'rd_type', 'runoff_c', 'erosion_c'], crs=self.model.crs)
        self.drain_points_gdf = gpd.GeoDataFrame(columns=['geometry', 'elevation'], crs=self.model.crs)
        self.transform = transform

        if isinstance(path, str):
            self.load(path)

    def reset_gdf(
        self,
    ):
        """
        Resets the geodataframe to an empty one
        """
        self.lines_gdf = gpd.GeoDataFrame(columns=['geometry', 'elevation', 'width', 'rd_type', 'runoff_c', 'erosion_c'], crs=self.model.crs)
        self.drain_points_gdf = gpd.GeoDataFrame(columns=['geometry', 'elevation'], crs=self.model.crs)
        return self

    def load(
        self,
        path: str,
    ):
        """
        Loads road data from a geospatial file, with automatic reprojection if CRS differs.

        Parameters:
        -----------
        path : string
            Path to the geospatial file
        """
        try:
            src = gpd.read_file(path)

            # CRS handling
            if src.crs is None:
                raise ValueError(f"Road file being attempted to load does not have a CRS.")
            elif self.model.crs is None:
                self.model.crs = src.crs
            elif src.crs is None:
                src.set_crs(self.model.crs)
            elif self.model.crs != src.crs:
                print(f"WARNING! Re-projecting!! CRS mismatch. Model CRS: {self.model.crs}, Source CRS: {src.crs}")
                src = src.to_crs(self.model.crs)

            self.lines_gdf = src
            self.bounds = self.lines_gdf.total_bounds

        except Exception as e:
            raise RuntimeError(f"Couldn't load the road data from {path}: {e}")

        return self

    def save(
        self, 
        path_name: str,
    ):
        """
        Saves road data to a geospatial file.

        Parameters:
        -----------
        path_name : string
            Name of output geopackage
        """
        try:
            self.lines_gdf.to_file(f'{path_name}.gpkg', layer='Road Layer', driver='GPKG', mode='w')
            self.drain_points_gdf.to_file(f'{path_name}.gpkg', layer='Drain Points', driver='GPKG', mode='w')

        except Exception as e:
            raise RuntimeError(f"Couldn't save the road data to {path_name}.gpkg: {e}")

        return self

    def plot(
        self,
    ):
        """
        Plot the road geometries.
        """

        self.lines_gdf.plot()
        self.drain_points_gdf.plot(ax=plt.gca(), color='red')

        return self

    def generate_advanced_road_pattern(
        self,
        exclusion_mask: np.ndarray | None = None,
        plot: bool = False,
        density: float | int = 0.042,
        target_slope: float = 0.15,
    ):
        raise NotImplementedError(
            "Method must be implemented in a subsequent cell."
        )

    def generate_simple_road_pattern(
        self,
        exclusion_mask: np.ndarray,
    ):
        r"""

        Parameters:
        -----------
        """

        raise NotImplementedError(
            "Method must be implemented in a subsequent cell."
        )

    def preprocess(
        self,
        target_segment_length: float = 2,
    ):
        r"""

        Parameters:
        -----------
        """

        raise NotImplementedError(
            "Method must be implemented in a subsequent cell."
        )

    def identify_local_min(
        self,
    ):

        r"""

        Parameters:
        -----------
        """

        raise NotImplementedError(
            "Method must be implemented in a subsequent cell."
        )
# |%%--%%| <gGD82UYvMp|QT8VivJDK1>
r"""°°°
Let's use what we learned to create a artifical elevation model that we can use to develop our model in. I want the environment to have three regions, each shaped like the objects in: \ _ / . We'll call this shape a valley. This shape is chosen to force drainage to the center.

[!NOTE]
Now that we've reviewed coordinate systems, you should be familiar with where the origin of our image is, in pixel coordinates or in geographical coordinates. So when I say north even while working with pixel coordinates, it should be intuitive that what I mean is the positive y direction.

West Flank (\\):
This region will represent a terrain that descends west to east with a linear slope gradient.

Base Region (_):
This region will depict a flat terrain with consistent elevation across its extent.

East Flank (/):
Mirror to the west flank, this area will be a terrain that ascends from west to east.
°°°"""
# |%%--%%| <QT8VivJDK1|4WZtPfJznC>

def generate_valley_elevation(
    self,
    width:int=100, height:int=100,
    flank_slope:float=0.1,
    base_region_elevation:Union[int,float]=0, base_region_percent:float=0.1,
    crs:int=6566, origin_x:float=320084.0, origin_y:float=251741.0,
    pixel_width:float=0.5, pixel_height:float=-0.5,
):
    r"""
    Create a Digital Elevation Model (DEM) with a terrain profile that looks like \_/, a valley.
    Creates a mask of the base region to be used in further processing steps.

    Parameters:
    -----------
    width : int
        Width of the DEM raster (default: 100)
    height : int
        Height of the DEM raster (default: 100)
    slope : float
        Slope gradient for flanks (default: 0.1)
    base_region_elevation : int or float
        Elevation of base region (default: 0)
    base_region_percent : float
        Percent of raster width used by base region (default: 0.1)
    crs : int
        EPSG Code as in integer. Note, if you update this, you also have to update origin_x/y for meaningful results. (default: 6566)
    origin_x : float
        X-coordinate of the raster origin - defualt created for CRS 6566 (default: 320084.0)
    origin_y : float
        Y-coordinate of the raster origin - default created for CRS 6566 (default: 251741.0)
    pixel_width : float
        Horizontal pixel resolution (default: 0.5)
    pixel_height : float
        Vertical pixel resolution (negative for top-down rasters) (default: -0.5)
    """

    # Initialize a zero-filled numpy array for the DEM
    dem = np.zeros((height, width), dtype=np.float32)

    # Calculate region widths for different terrain sections
    if base_region_percent > 1: raise ValueError(f"Percent greater than 1 is invalid: {base_region_percent}")
    west_flank_width = east_flank_width = int(width * (1-base_region_percent)/2)
    base_region_width = width - west_flank_width - east_flank_width

    # Create a mask over the base region so we can keep it draining
    exclusion_mask = np.zeros((height, width), dtype=bool); exclusion_mask[:, west_flank_width:-east_flank_width] = True
    self.exclusion_mask = exclusion_mask

    # Calculate slope height based on region width and slope parameter
    slope_height = west_flank_width * pixel_width * flank_slope

    # Create terrain profile for west flank (descending elevation)
    west_flank_row = np.linspace(slope_height, 0, west_flank_width)

    # Create terrain profile for base region
    base_region_row = np.full(base_region_width, base_region_elevation)

    # Create terrain profile for wast flank (ascending elevation)
    east_flank_row = west_flank_row[::-1]

    # Combine terrain sections into a single row
    single_row = np.concatenate([
        west_flank_row, 
        base_region_row, 
        east_flank_row
    ])

    # Replicate the single row to create the full DEM
    array = np.tile(single_row, (height, 1))

    # Create geographic transform for spatial referencing
    transform = (
        pixel_width,     # x-axis pixel width (horizontal resolution)
        0.00,            # x-axis rotation/skew
        origin_x,        # x-axis translation (easting origin)
        0.00,            # y-axis rotation/skew
        pixel_height,    # y-axis pixel height (vertical resolution)
        origin_y         # y-axis translation (northing origin)
    )

    # Define rasterio profile for GeoTIFF creation
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': dem.dtype,
        'crs': pyproj.CRS.from_user_input(crs),
        'transform': transform,
        'nodata': -9999,
    }

    # Make sure the raster is able to be written
    try:
        temp_output_path = BytesIO()
        with rasterio.open(temp_output_path, 'w', **profile) as dst:
            self.array = array
            self.profile = dst.profile
            self.model.crs = dst.crs
            self.transform = dst.transform
            self.bounds = dst.bounds
            self.nodata = dst.nodata

            self.model.roads.lines_gdf = self.model.roads.lines_gdf.set_crs(dst.crs) # Usually handled by load functions
    except Exception as e:
        raise RuntimeError(f"Something went wrong when trying to create geotiff: {e}")
    finally:
        if isinstance(temp_output_path, BytesIO):
            temp_output_path.close()

    return self

Elevation.generate_valley_elevation = generate_valley_elevation
# |%%--%%| <4WZtPfJznC|wUuT4JAImG>
r"""°°°
Let's generate and plot a 500x500px terrain with a 20% slope gradient!
°°°"""
# |%%--%%| <wUuT4JAImG|BlZznEs9Qu>

# Initialize our model class
m = Model()

# Generate and plot elevation
m.elevation.generate_valley_elevation(width=500,height=500,flank_slope=0.20).plot(cmap="terrain")

# |%%--%%| <BlZznEs9Qu|gOB0kT61nE>
r"""°°°
Alright! Now, let's add some local features. This will serve the purpose of forcing flowpaths to travel around some feature.

We're going to do this by generating height values from a unit Gaussian distribution.
°°°"""
# |%%--%%| <gOB0kT61nE|H0LCVkQh9f>

def generate_local_features(
    self,
    exclusion_mask: np.ndarray | None = None,
    gauss_norm_distribution_mu=0, gauss_norm_distribution_sigma=0.5, 
    gauss_smoothness_filter_sigma=3, 
    should_apply_gaussian_smoothing = True, smooth_exclusion_regions = False,
    grid_shape=(10,10)):
    r"""

    Parameters:
    -----------
    exclusion_mask : numpy.ndarray of dtype Bool, optional
        Mask area to not add any noise (optionally still include in smoothing, check `smooth_exclusion_regions`)
    gauss_norm_distribution_mu : float
        Controls the mean elevation of the raster
    gauss_norm_distribution_sigma : float
        Controls the amplitude of terrain variations
    gauss_smoothness_filter_sigma : float
        Controls the smoothness of features
    should_apply_gaussian_smoothing : Boolean
        Should the features be smoothed (default = True)
    smooth_exclusion_regions : Boolean
        If a exclusion mask is provided, should this area be smoothed to blend features in (defualt = False)
    grid_shape : (int,int)
        Controls the dimensions of local features (default = (10,10))

    Notes:
    ------
    The method generates terrain variations by:
    1. Creating a sparse grid of random height values
    2. Interpolating between these points smoothly
    3. Applying Gaussian smoothing for gentle transitions
    """

    height, width = self.array.shape

    # Create a sparse noise grid
    grid_height = height // grid_shape[0]
    grid_width = width // grid_shape[1]

    noise_grid = np.random.normal(
        gauss_norm_distribution_mu,
        gauss_norm_distribution_sigma,
        size=(grid_height, grid_width))

    noise_grid = np.maximum(noise_grid, 0)

    # Create coordinate arrays for interpolation
    y_sparse_coords = np.linspace(0, height-1, grid_height)
    x_sparse_coords = np.linspace(0, width-1, grid_width)

    # Create full pixel coordinate arrays
    y_pixel_coords = np.arange(height)
    x_pixel_coords = np.arange(width)

    # Interpolation values between the grid values
    interpolator = scipy.interpolate.RectBivariateSpline(
        y_sparse_coords, 
        x_sparse_coords, 
        noise_grid
    )
    interpolated_noise =  interpolator(y_pixel_coords, x_pixel_coords)

    if isinstance(exclusion_mask, np.ndarray):
        interpolated_noise[exclusion_mask] = 0

    array = self.array + interpolated_noise

    if should_apply_gaussian_smoothing:
        if exclusion_mask is None or smooth_exclusion_regions:
            array = scipy.ndimage.gaussian_filter(array, sigma=gauss_smoothness_filter_sigma)
        elif isinstance(exclusion_mask, np.ndarray) and not smooth_exclusion_regions:
            array = np.where(
                exclusion_mask, 
                self.array, 
                scipy.ndimage.gaussian_filter(array, sigma=gauss_smoothness_filter_sigma)
            )

    self.array = array

    return self

Elevation.generate_local_features = generate_local_features

# |%%--%%| <H0LCVkQh9f|aWXIr6nRoi>
r"""°°°
Phew, that was a lot of work! With just these two functions, we now have the freedom to generate a wide range of terrains. I suggest taking some time to play around with all the options to get comfortable with what kind of terrains you're able to create. It's pretty easy to create impossible terrains if you're not careful!
°°°"""
# |%%--%%| <aWXIr6nRoi|g9Ej92vRLu>

m = Model()

m.elevation.generate_valley_elevation(width=200, height=200, flank_slope=0.2, base_region_percent=0.1, base_region_elevation=0).plot()

m.elevation.generate_local_features(
        exclusion_mask=m.elevation.exclusion_mask,
        gauss_norm_distribution_mu=20, gauss_norm_distribution_sigma=1,
        gauss_smoothness_filter_sigma=1, should_apply_gaussian_smoothing=True, smooth_exclusion_regions=False,
        grid_shape=(20,20)
).save('testing.tif').plot3d()

#|%%--%%| <g9Ej92vRLu|jOgZwYSx76>

m.elevation.generate_valley_elevation(width=1000, height=1000, flank_slope=0.2, base_region_percent=0.1, base_region_elevation=-10).plot()

m.elevation.generate_local_features(
        exclusion_mask=m.elevation.exclusion_mask,
        gauss_norm_distribution_mu=20, gauss_norm_distribution_sigma=20,
        gauss_smoothness_filter_sigma=3, should_apply_gaussian_smoothing=False, smooth_exclusion_regions=False,
        grid_shape=(200,200)
).save('testing.tif').plot3d()


# |%%--%%| <jOgZwYSx76|nfsFdtslzH>
r"""°°°
Now, this model *is* called RoadConnect, so let's get just road types added!
°°°"""
#|%%--%%| <nfsFdtslzH|frJ8i6bXx6>

m.roads.types = [
    RoadParametersStruct(name='Test1', runoff_coeff=0.9, erosion_coeff=0.75),
    RoadParametersStruct(name='Test2', runoff_coeff=0.3, erosion_coeff=2.0),
    RoadParametersStruct(name='Test3', runoff_coeff=0.5, erosion_coeff=0.1)
]

#|%%--%%| <frJ8i6bXx6|W0fFrX4THB>
r"""°°°

NOTE: This `local minima and maxima` section was added when I had a planned use that no longer exists. However, there is still good informaiton here so I'll keep this section.

To find local minima and maxima, we'll first create a binary structure that defines our filter footprint. What the heck is that and how do we make one? A binary structure is an array of boolean values that defines a pattern for morphological operations, serving as a footprint to specify which neighboring cells should be compared to a center cell when applying our max or min filters. This can be easily created using `scipy.ndimage`. The function used to create this structure also requires a `rank` parameter, which defines the number of dimensions your data has. You might think to set this to three because our data is 3-dimensional, right? Wrong. Our data can be practically classified as 2.5D but it is fundamentally two-dimensional as it represents elevation data in a two-dimensional grid (run `m.model.array.shape`). If we'd been working with point-cloud data, this would have been set to three. The other parameter accepted by this function is `connectivity`, and in a 2-D grid, there's two options (`connectivity` can range from 1 to the number of dimensions you have):

Diagonals are not considered:

[[False,  True, False],
 [ True,  True,  True],
 [False,  True, False]]

or, diagonals are considered:

[[ True,  True,  True],
 [ True,  True,  True],
 [ True,  True,  True]].

For our search, we're going to include diagonals.

Another valid approach of doing this is to instead supply a size to the max/min filter (and considering it's less code in this case, you may even prefer it). Both these options, however, (can) restrict us to a rectangular footprint (I say "can" because without diagonals, the binary structure is irregular). What if we wanted to describe a complex irregular footprint to apply our filter? Well, learning from our usage of the binary structure, we can create one from scratch!

[[False, False, True,  False, False],
 [False, True,  True,   True, False],
 [True,  True,  False,  True,  True],
 [False, False, True,  False, False]]

Our use case doesn't necessitate usage of an irregular shape but this was simply an exercise in learning about the tools available to us.

The filter will return an array of `True` or `False` values that we can find the coordinates of by using `np.where` and then join the `x` and `y` coordinates using np.column_stack.

°°°"""
# |%%--%%| <W0fFrX4THB|yQCxcTg45c>

def identify_local_min_max(
    self,
    footprint: np.ndarray = scipy.ndimage.generate_binary_structure(2, 2),
    exclusion_mask: np.ndarray | None = None,
    plot: bool = False
):
    r"""

    Parameters:
    -----------
    footprint : np.ndarry
        Footprint over which to apply min/max filter (default: 3x3)
    exclusion_mask : np.ndarray
        Don't identify any min/max in this region
    """
    
    if isinstance(exclusion_mask,np.ndarray):
        local_max = (self.array == scipy.ndimage.maximum_filter(m.elevation.array, footprint=footprint)) & (~exclusion_mask)
        local_min = (self.array == scipy.ndimage.minimum_filter(m.elevation.array, footprint=footprint)) & (~exclusion_mask)
    else:
        local_max = (self.array == scipy.ndimage.maximum_filter(m.elevation.array, footprint=footprint))
        local_min = (self.array == scipy.ndimage.minimum_filter(m.elevation.array, footprint=footprint))

    self.local_max_coords = list(zip(*np.where(local_max)))
    self.local_min_coords = list(zip(*np.where(local_min)))

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title('Local Maxima Points')
        plt.imshow(m.elevation.array, cmap='viridis')
        plt.scatter(
            [coord[1] for coord in self.local_max_coords], 
            [coord[0] for coord in self.local_max_coords], 
            color='red', 
            s=10, 
            alpha=0.7
        )

        plt.subplot(122)
        plt.title('Local Minima Points')
        plt.imshow(m.elevation.array, cmap='viridis')
        min_coords = np.column_stack(np.where(local_min))
        plt.scatter(
            [coord[1] for coord in self.local_min_coords], 
            [coord[0] for coord in self.local_min_coords], 
            color='blue', 
            s=10, 
            alpha=0.7
        )

        plt.tight_layout()
        plt.show()


    return self

Elevation.identify_local_min_max = identify_local_min_max

#|%%--%%| <yQCxcTg45c|CqpYU4hLjx>

edge_mask = np.zeros_like(m.elevation.exclusion_mask, dtype=bool)

ignore_edge_width = 4

edge_mask[:ignore_edge_width, :] = True

edge_mask[-ignore_edge_width:, :] = True

edge_mask[:, :ignore_edge_width] = True

edge_mask[:, -ignore_edge_width:] = True

# Combine the existing exclusion mask with the new edge mask
combined_exclusion_mask = m.elevation.exclusion_mask | edge_mask

m.elevation.identify_local_min_max(
    exclusion_mask=combined_exclusion_mask,
    plot=True
)

#|%%--%%| <CqpYU4hLjx|pfsdZH1led>
r"""°°°
We have road types added, but no road segments. To develop our model, we'll create a simple road pattern: for each road type, we'll generate a road that runs west to east across every region in the raster.

Remember all that time we spent learning about transforms? Those skills will be crucial here because we'll be:

>> Creating a new coordinate system for each region
>> Converting back to geographic coordinates using two transforms

This process requires extra attention due to the different transform implementations we're using:

>> raster.io uses it's own `affine.Affine` class to represent the transform
>> geopandas uses `shapely.affinity.affine_transform`

The key challenge is that these libraries represent the transform matrix differently in one-dimensional space. We'll need to carefully re-create the transform to ensure accurate coordinate conversion.
°°°"""
#|%%--%%| <pfsdZH1led|x7W9peFIHL>

def generate_simple_road_pattern(
    self,
    exclusion_mask: np.ndarray,
):
    r"""

    Parameters:
    -----------
    """

    #TODO: This implementaion doesn't work without an exlusion mask

    labeled_array, num_features = scipy.ndimage.label(np.logical_not(exclusion_mask))

    regions = [
        [
            self.model.elevation.array[
                np.any(labeled_array == label, axis=1).nonzero()[0].min():
                np.any(labeled_array == label, axis=1).nonzero()[0].max() + 1,
                np.any(labeled_array == label, axis=0).nonzero()[0].min():
                np.any(labeled_array == label, axis=0).nonzero()[0].max() + 1
            ],
            # Modified affine transform for the cropped region
            [ # [a, b, d, e, xoff, yoff] shapely.affinity.affine_transform
                1, 0, 0, 1,
                np.any(labeled_array == label, axis=0).nonzero()[0].min(),
                np.any(labeled_array == label, axis=1).nonzero()[0].min()
            ]
        ]
        for label in range(1, num_features + 1)
    ]

    for region, transform in regions:

        height, width = region.shape

        x_buffer = width * 0.1
        y_buffer = height * 0.1

        usable_width = width - (2 * x_buffer)
        usable_height = height - (2 * y_buffer)

        num_types = len(self.types)
        vertical_spacing = usable_height / (num_types + 1)

        for i, road_type in enumerate(self.types):
            y_pos = y_buffer + (i + 1)  * vertical_spacing

            # Create a line from left to right
            line_coords = [
                (x_buffer, y_pos), 
                (width - x_buffer, y_pos)
            ]

            line = shapely.geometry.LineString(line_coords) # Create a line with region coords
            line = shapely.affinity.affine_transform(line, transform) # Transform line to use coords of elevation raster
            line = shapely.affinity.affine_transform(line, [ # Transform line to use geocoords
                self.model.elevation.transform.a,
                self.model.elevation.transform.b,
                self.model.elevation.transform.d,
                self.model.elevation.transform.e,
                self.model.elevation.transform.c,
                self.model.elevation.transform.f,
            ])

            self.lines_gdf = self.lines_gdf._append({
                'geometry': line,
                #'elevation': None, # No elevation yet because we need to segmentize the roads
                'rd_type': road_type.name,
                'runoff_c': road_type.runoff_coeff,
                'erosion_c': road_type.erosion_coeff,
                'width': 3,
            }, ignore_index=True)


    self.lines_gdf = self.lines_gdf.set_crs(self.model.crs)

    return self

Road.generate_simple_road_pattern = generate_simple_road_pattern

#|%%--%%| <x7W9peFIHL|k470527plJ>

m.roads.reset_gdf().generate_simple_road_pattern(
    scipy.ndimage.binary_dilation(m.elevation.exclusion_mask, iterations=10) # Dilate to prevent roads from being right up next to the edge of the center
).plot()

#|%%--%%| <k470527plJ|qRUbsXNXgw>
r"""°°°
Next, we'll need to do some preprocessing on our road network. To understand why I made the decisions I did here, let's quickly discuss how some future parts of the model will work.

To figure out how many meteres of road are connected to a certain drain point, we're going to trace a path from each road segment down the elevation gradient until we reach a drain point. These drain points will be automatically created at first based on road elevation values but can be modified in intermediate steps if it is determined that some points are extraneous or due to noisy data (there's another project in-progress to help solve this issue that will be attached to the end). 

Now, preprocessing! It wouldn't be feasible or reasonable to trace every centimeter of road surface to a drain point so let's first simplify and segmentize our road to ~2m segments that will serve as the base for all future computations. 

After this is done, we're going to assign elevation values to the road segments by sampling the elevation raster. This section currently implements a basic elevation calculation, but as mentioned before, I plan on making this more resistant to noise via some more advanced sampling methods in the future.
°°°"""
#|%%--%%| <qRUbsXNXgw|0mT7qQbOcD>

def preprocess(
    self,
    target_segment_length: float = 2,
):
    r"""

    Parameters:
    -----------
    """

    segmentized_data = []

    #rd_gdal = rd.LoadGDAL('testing.tif')
    #rd_elevation = rd.rdarray(m.elevation.array, no_data=m.elevation.nodata)
    #rd_filled = rd.FillDepressions(rd_gdal, epsilon=True, in_place=False)
    #rd.SaveGDAL('resolvedflats.tif', rd_filled)

    for _, row in self.lines_gdf.iterrows():
        total_length = row.geometry.length
        num_segments = max(1, round(total_length / target_segment_length))

        for i in range(num_segments):
            start_point = row.geometry.interpolate(i / num_segments, normalized=True)
            end_point = row.geometry.interpolate((i + 1) / num_segments, normalized=True)

            segment = shapely.geometry.LineString([start_point, end_point])

            new_row = row.copy()
            new_row['geometry'] = segment

            elevation_stats = rasterstats.zonal_stats(
                segment.buffer(0.1, cap_style='flat'),
                m.elevation.array,
                affine=self.model.elevation.transform,
                stats=['mean'],
                all_touched=True,
                nodata=self.model.elevation.nodata
            )

            new_row['elevation'] = elevation_stats[0]['mean']

            segmentized_data.append(new_row)

    self.lines_gdf = gpd.GeoDataFrame(segmentized_data, crs=self.model.crs)

    return self

Road.preprocess = preprocess

#|%%--%%| <0mT7qQbOcD|G51hQa0nMJ>

m.roads.reset_gdf().generate_simple_road_pattern(
    scipy.ndimage.binary_dilation(m.elevation.exclusion_mask, iterations=10) # Dilate to prevent roads from being right up next to the edge of the center
).preprocess().plot()

#|%%--%%| <G51hQa0nMJ|MzOLm6yezl>

def identify_local_min(
    self,
):

    r"""

    Parameters:
    -----------
    """

    lowest_elevation_points = []

    for _, row in self.lines_gdf.iterrows():
        touching_lines = self.lines_gdf[self.lines_gdf.geometry.touches(row.geometry)]

        is_lowest = all(row['elevation'] <= touch_row.elevation 
                        for touch_row in touching_lines.itertuples(index=False))

        if is_lowest:
            center_point = row.geometry.centroid
            lowest_elevation_points.append({
                'geometry': center_point,
                'elevation': row['elevation']
            })

    self.drain_points_gdf = gpd.GeoDataFrame(
        lowest_elevation_points,
        geometry='geometry',
        crs=self.model.crs
    )

    return self

Road.identify_local_min = identify_local_min

#|%%--%%| <MzOLm6yezl|OV9Qt9L2y9>

m.roads.identify_local_min().plot().save('testpattern')

#|%%--%%| <OV9Qt9L2y9|BVKPh9kEm6>

m = Model()

m.elevation.load('dem_0.5.tif')

#|%%--%%| <BVKPh9kEm6|YMtIq4smd9>

m.roads.reset_gdf().load('road_edges_modified.shp').preprocess().identify_local_min().save('testpattern')

#|%%--%%| <YMtIq4smd9|BKlpUvmeYG>

import richdem as rd
from pysheds.grid import Grid

rd_elevation = rd.rdarray(m.elevation.array, no_data=m.elevation.nodata)
rd_filled = rd.FillDepressions(rd_elevation, epsilon=True, in_place=False)
flow_proportions = rd.FlowProportions(rd_filled, method='D8')

def trace_flow_path(flow_proportions, start_indices, road_gdf=None, transform=None):
    # Neighbor indices corresponding to the layout you described
    # 0th value is cell status
    # Neighbors in order: left, top-left, top, top-right, right, bottom-right, bottom, bottom-left
    neighbor_mapping = {
        'left': 1,
        'top-left': 2,
        'top': 3,
        'top-right': 4,
        'right': 5,
        'bottom-right': 6,
        'bottom': 7,
        'bottom-left': 8
    }
    
    # Initialize path with the starting point
    path = [start_indices]
    current_indices = start_indices
    
    max_iterations = flow_proportions.shape[0] * flow_proportions.shape[1]  # Prevent infinite loops
    iterations = 0
    
    while iterations < max_iterations:
        # Get the current cell's flow proportions
        current_cell = flow_proportions[current_indices[0], current_indices[1]]
        
        # Check cell status (0th value)
        if current_cell[0] == -2:  # NoData cell
            break
        elif current_cell[0] == -1:  # No flow cell
            break
        
        # Find the maximum flow proportion among valid neighbors
        max_flow = 0
        next_indices = None
        chosen_neighbor = None
        
        # Check each neighbor in the specified order
        for neighbor, idx in neighbor_mapping.items():
            flow_prop = current_cell[idx]
            
            # If this flow proportion is larger than current max
            if flow_prop > max_flow:
                max_flow = flow_prop
                chosen_neighbor = neighbor
        
        # If no flow found, break
        if chosen_neighbor is None:
            break
        
        # Determine next indices based on the chosen neighbor
        # Remember: top-left is the origin (0,0), so:
        # - Decreasing row goes up
        # - Increasing row goes down
        # - Decreasing column goes left
        # - Increasing column goes right
        if chosen_neighbor == 'left':
            next_indices = (current_indices[0], current_indices[1] - 1)
        elif chosen_neighbor == 'top-left':
            next_indices = (current_indices[0] - 1, current_indices[1] - 1)
        elif chosen_neighbor == 'top':
            next_indices = (current_indices[0] - 1, current_indices[1])
        elif chosen_neighbor == 'top-right':
            next_indices = (current_indices[0] - 1, current_indices[1] + 1)
        elif chosen_neighbor == 'right':
            next_indices = (current_indices[0], current_indices[1] + 1)
        elif chosen_neighbor == 'bottom-right':
            next_indices = (current_indices[0] + 1, current_indices[1] + 1)
        elif chosen_neighbor == 'bottom':
            next_indices = (current_indices[0] + 1, current_indices[1])
        elif chosen_neighbor == 'bottom-left':
            next_indices = (current_indices[0] + 1, current_indices[1] - 1)
        
        # Prevent revisiting the same cell
        if next_indices in path:
            break
        
        # Ensure next indices are within array bounds
        if (next_indices[0] < 0 or next_indices[0] >= flow_proportions.shape[0] or
            next_indices[1] < 0 or next_indices[1] >= flow_proportions.shape[1]):
            break
        
        # Add next point to path
        path.append(next_indices)
        current_indices = next_indices
        
        iterations += 1
        
        # Optional: break if max iterations reached
        if iterations >= max_iterations:
            break
    
    return path

# Modified trace_and_save_flow_paths function

def trace_and_save_flow_paths(points_gdf, flow_proportions, m):
    # Prepare to store all paths and their transformed coordinates
    all_paths = []
    
    # Will store geometries for the shapefile
    path_geometries = []
    
    # Trace path for each point
    for idx, point in points_gdf.iterrows():
        # Transform point coordinates to pixel indices
        current_indices = ~m.elevation.transform * (point.geometry.x, point.geometry.y)
        current_indices = (int(round(current_indices[1])), int(round(current_indices[0])))
        
        # Trace the flow path

def trace_and_save_flow_paths(points_gdf, flow_proportions, m):
    # Prepare to store all paths and their transformed coordinates
    all_paths = []
    
    # Will store geometries for the shapefile
    path_geometries = []
    
    # Get cell size from the transform
    cell_size = abs(m.elevation.transform[0])
    
    # Trace path for each point
    for idx, point in points_gdf.iterrows():
        # Transform point coordinates to pixel indices
        current_indices = ~m.elevation.transform * (point.geometry.x, point.geometry.y)
        current_indices = (int(round(current_indices[1])), int(round(current_indices[0])))
        
        # Trace the flow path
        flow_path = trace_flow_path(
            flow_proportions, 
            current_indices, 
            road_gdf=m.roads.lines_gdf,  # Assuming m.roads_gdf exists
            transform=m.elevation.transform  # Pass the transform
        )
        
        # Skip empty paths
        if len(flow_path) < 3:  # Ensure we have at least 3 cells to work with
            continue
        
        all_paths.append(flow_path)
        
        # Transform path indices back to geographic coordinates
        geo_path = []
        for path_idx in flow_path:
            # Use the forward transform to get geographic coordinates
            geo_x, geo_y = m.elevation.transform * (path_idx[1], path_idx[0])
            geo_path.append((geo_x, geo_y))
        
        from shapely.geometry import LineString, Point
        
        # Start with the third point in the path
        modified_geo_path = geo_path[2:]
        
        # Insert the original point at the beginning
        modified_geo_path.insert(0, (point.geometry.x, point.geometry.y))
        
        # Create LineString with the modified path
        path_line = LineString(modified_geo_path)
        path_geometries.append(path_line)
    
    # Create a GeoDataFrame from the path geometries
    paths_gdf = gpd.GeoDataFrame(geometry=path_geometries, crs=points_gdf.crs)
    
    # Save to shapefile
    paths_gdf.to_file('flow_paths.shp')
    
    return all_paths, paths_gdf


all_paths, paths_gdf = trace_and_save_flow_paths(
    points_gdf=m.roads.drain_points_gdf,  # GeoDataFrame of starting points
    flow_proportions=flow_proportions,    # Your flow proportions array
    m=m                                   # The map object containing transforms and road data
)

#|%%--%%| <BKlpUvmeYG|AQ5x7IOePF>


import geopandas as gpd
import shapely
import shapely.geometry
import shapely.ops

def segment_line(geometry, target_length):
    """
    Segment a line into approximately equal-length segments.
    
    :param geometry: Input line geometry
    :param target_length: Desired segment length in meters
    :return: List of line segments
    """
    total_length = geometry.length
    num_segments = max(1, round(total_length / target_length))
    
    segments = []
    for i in range(num_segments):
        start_point = geometry.interpolate(i / num_segments, normalized=True)
        end_point = geometry.interpolate((i + 1) / num_segments, normalized=True)
        
        segment = shapely.geometry.LineString([start_point, end_point])
        segments.append(segment)
    
    return segments

def find_drain_point(flow_path, drain_points):
    """
    Find the corresponding drain point for a given flow path.
    
    :param flow_path: Flow path geometry
    :param drain_points: GeoDataFrame of drain points
    :return: Drain point geometry or None
    """
    for _, drain_point in drain_points.iterrows():
        if drain_point.geometry.intersects(flow_path.geometry):
            return drain_point.geometry
    return None

def trim_flow_path(flow_path, drain_points, road_lines, road_sindex, target_segment_length=2):
    """
    Trim a single flow path based on road intersections.
    
    :param flow_path: Original flow path
    :param drain_points: GeoDataFrame of drain points
    :param road_lines: GeoDataFrame of road lines
    :param road_sindex: Spatial index of road lines
    :param target_segment_length: Desired segment length in meters
    :return: Trimmed flow path or None
    """
    # Find the starting drain point
    start_point = find_drain_point(flow_path, drain_points)
    if start_point is None:
        print(f"No matching drain point found for flow path")
        return None

    # Segment the flow path
    segmented_path = segment_line(flow_path.geometry, target_segment_length)
    
    # Trim the path
    trimmed_segments = []
    last_point = start_point.coords[0]
    intersected = False
    
    # Skip the first segment, but start from the second segment's end point
    for segment in segmented_path[1:]:
        # Create a connecting segment from the last point
        connecting_segment = shapely.geometry.LineString([last_point, segment.coords[-1]])
        
        # Find potential road intersections using spatial index
        possible_matches_index = list(road_sindex.intersection(segment.bounds))
        
        road_intersection = False
        for road_idx in possible_matches_index:
            road = road_lines.iloc[road_idx]
            if segment.intersects(road.geometry):
                # Find the intersection point
                intersection = segment.intersection(road.geometry)
                
                # Create trimmed path segments
                trimmed_segments.append(connecting_segment)
                final_segment = shapely.geometry.LineString([
                    connecting_segment.coords[-1], 
                    intersection
                ])
                trimmed_segments.append(final_segment)
                
                road_intersection = True
                intersected = True
                break
        
        # If no intersection, add the connecting segment and current segment
        if not road_intersection:
            trimmed_segments.append(connecting_segment)
            trimmed_segments.append(segment)
            last_point = segment.coords[-1]
        
        # Stop if we've found an intersection
        if intersected:
            break
    
    # Combine segments into a single LineString
    if trimmed_segments:
        try:
            # Combine all segments into a single LineString
            combined_line = shapely.ops.linemerge(trimmed_segments)
            
            # Create a new GeoDataFrame row
            trimmed_flow_path = flow_path.copy()
            trimmed_flow_path['geometry'] = combined_line
            return trimmed_flow_path
        except Exception as e:
            print(f"Error merging segments: {e}")
            return None
    
    return None

def trim_flow_paths(flow_paths, drain_points, road_lines, target_segment_length=2):
    """
    Trim multiple flow paths.
    
    :param flow_paths: GeoDataFrame of flow paths
    :param drain_points: GeoDataFrame of drain points
    :param road_lines: GeoDataFrame of road lines
    :param target_segment_length: Desired segment length in meters
    :return: GeoDataFrame of trimmed flow paths
    """
    # Create a spatial index for road lines
    road_sindex = road_lines.sindex
    
    # Process each flow path
    trimmed_flow_paths = []
    
    for _, flow_path in flow_paths.iterrows():
        trimmed_path = trim_flow_path(
            flow_path, 
            drain_points, 
            road_lines, 
            road_sindex, 
            target_segment_length
        )
        
        if trimmed_path is not None:
            trimmed_flow_paths.append(trimmed_path)
    
    # Create a new GeoDataFrame with trimmed flow paths
    return gpd.GeoDataFrame(trimmed_flow_paths, crs=flow_paths.crs)

def process_flow_paths(m, output_path='flow_paths_trimmed.shp', target_segment_length=2):
    """
    Main processing function to trim flow paths.
    
    :param m: Module or object containing road-related data
    :param output_path: Path to save trimmed flow paths
    :param target_segment_length: Desired segment length in meters
    :return: GeoDataFrame of trimmed flow paths
    """
    # Extract necessary data
    drain_points = m.roads.drain_points_gdf
    road_lines = m.roads.lines_gdf
    flow_paths = gpd.read_file('flow_paths.shp')
    
    # Trim flow paths
    trimmed_flow_paths_gdf = trim_flow_paths(
        flow_paths, 
        drain_points, 
        road_lines, 
        target_segment_length
    )
    
    # Save to shapefile
    trimmed_flow_paths_gdf.to_file(output_path)
    
    return trimmed_flow_paths_gdf


result = process_flow_paths(m)

#|%%--%%| <AQ5x7IOePF|wUCU0y5zsq>


import richdem as rd
import geopandas as gpd
import shapely.geometry
import shapely.ops

def process_flow_paths(m, target_segment_length=2):
    """
    Comprehensive flow path processing function.
    
    :param m: Module containing elevation, roads, and other geospatial data
    :param target_segment_length: Desired segment length in meters
    :return: GeoDataFrame of trimmed flow paths
    """
    # 1. Prepare elevation data for flow calculations
    rd_elevation = rd.rdarray(m.elevation.array, no_data=m.elevation.nodata)
    rd_filled = rd.FillDepressions(rd_elevation, epsilon=True, in_place=False)
    flow_proportions = rd.FlowProportions(rd_filled, method='D8')

    def trace_flow_path(flow_proportions, start_indices):
        """
        Trace flow path based on flow proportions.
        
        :param flow_proportions: Flow proportion array
        :param start_indices: Starting pixel indices
        :return: List of pixel indices representing the flow path
        """
        # Neighbor indices mapping (similar to previous implementation)
        neighbor_mapping = {
            'left': 1, 'top-left': 2, 'top': 3, 'top-right': 4,
            'right': 5, 'bottom-right': 6, 'bottom': 7, 'bottom-left': 8
        }
        
        path = [start_indices]
        current_indices = start_indices
        
        max_iterations = flow_proportions.shape[0] * flow_proportions.shape[1]
        iterations = 0
        
        while iterations < max_iterations:
            current_cell = flow_proportions[current_indices[0], current_indices[1]]
            
            # Check cell status
            if current_cell[0] in [-2, -1]:  # NoData or No flow cell
                break
            
            # Find maximum flow proportion
            max_flow = 0
            chosen_neighbor = None
            
            for neighbor, idx in neighbor_mapping.items():
                flow_prop = current_cell[idx]
                if flow_prop > max_flow:
                    max_flow = flow_prop
                    chosen_neighbor = neighbor
            
            # Determine next indices
            if chosen_neighbor is None:
                break
            
            # Mapping of neighbors to index changes
            neighbor_deltas = {
                'left': (0, -1), 'top-left': (-1, -1), 'top': (-1, 0),
                'top-right': (-1, 1), 'right': (0, 1), 'bottom-right': (1, 1),
                'bottom': (1, 0), 'bottom-left': (1, -1)
            }
            
            delta = neighbor_deltas[chosen_neighbor]
            next_indices = (current_indices[0] + delta[0], current_indices[1] + delta[1])
            
            # Bounds and revisit checks
            if (next_indices[0] < 0 or next_indices[0] >= flow_proportions.shape[0] or
                next_indices[1] < 0 or next_indices[1] >= flow_proportions.shape[1] or
                next_indices in path):
                break
            
            path.append(next_indices)
            current_indices = next_indices
            iterations += 1
        
        return path

    def trace_and_save_flow_paths(points_gdf, flow_proportions, m):
        """
        Trace flow paths for multiple points and save as shapefile.
        
        :param points_gdf: GeoDataFrame of starting points
        :param flow_proportions: Flow proportion array
        :param m: Map object with transforms
        :return: Tuple of all paths and GeoDataFrame of path geometries
        """
        all_paths = []
        path_geometries = []
        
        for idx, point in points_gdf.iterrows():
            # Transform point coordinates to pixel indices
            current_indices = ~m.elevation.transform * (point.geometry.x, point.geometry.y)
            current_indices = (int(round(current_indices[1])), int(round(current_indices[0])))
            
            # Trace the flow path
            flow_path = trace_flow_path(flow_proportions, current_indices)
            
            # Skip empty paths
            if len(flow_path) < 3:
                continue
            
            all_paths.append(flow_path)
            
            # Transform path indices back to geographic coordinates
            geo_path = []
            for path_idx in flow_path:
                geo_x, geo_y = m.elevation.transform * (path_idx[1], path_idx[0])
                geo_path.append((geo_x, geo_y))
            
            # Create LineString with the modified path
            modified_geo_path = geo_path[2:]
            modified_geo_path.insert(0, (point.geometry.x, point.geometry.y))
            
            path_line = shapely.geometry.LineString(modified_geo_path)
            path_geometries.append(path_line)
        
        # Create a GeoDataFrame from the path geometries
        paths_gdf = gpd.GeoDataFrame(geometry=path_geometries, crs=points_gdf.crs)
        paths_gdf.to_file('flow_paths.shp')
        
        return all_paths, paths_gdf

    def segment_line(geometry, target_length):
        """
        Segment a line into approximately equal-length segments.
        
        :param geometry: Input line geometry
        :param target_length: Desired segment length in meters
        :return: List of line segments
        """
        total_length = geometry.length
        num_segments = max(1, round(total_length / target_length))
        
        segments = []
        for i in range(num_segments):
            start_point = geometry.interpolate(i / num_segments, normalized=True)
            end_point = geometry.interpolate((i + 1) / num_segments, normalized=True)
            
            segment = shapely.geometry.LineString([start_point, end_point])
            segments.append(segment)
        
        return segments

    def trim_flow_paths(flow_paths, drain_points, road_lines, target_segment_length=2):
        """
        Trim flow paths based on road intersections.
        
        :param flow_paths: GeoDataFrame of flow paths
        :param drain_points: GeoDataFrame of drain points
        :param road_lines: GeoDataFrame of road lines
        :param target_segment_length: Desired segment length in meters
        :return: GeoDataFrame of trimmed flow paths
        """
        # Create spatial index for road lines
        road_sindex = road_lines.sindex
        trimmed_flow_paths = []

        def find_drain_point(flow_path):
            """Find the corresponding drain point for a given flow path."""
            for _, drain_point in drain_points.iterrows():
                if drain_point.geometry.intersects(flow_path.geometry):
                    return drain_point.geometry
            return None

        def trim_single_flow_path(flow_path):
            """Trim a single flow path based on road intersections."""
            # Find the starting drain point
            start_point = find_drain_point(flow_path)
            if start_point is None:
                return None

            # Segment the flow path
            segmented_path = segment_line(flow_path.geometry, target_segment_length)
            
            # Trim the path
            trimmed_segments = []
            last_point = start_point.coords[0]
            intersected = False

            for segment in segmented_path[1:]:
                # Create a connecting segment
                connecting_segment = shapely.geometry.LineString([last_point, segment.coords[-1]])
                
                # Find potential road intersections

                possible_matches_index = list(road_sindex.intersection(segment.bounds))
                
                road_intersection = False
                for road_idx in possible_matches_index:
                    road = road_lines.iloc[road_idx]
                    if segment.intersects(road.geometry):
                        # Find the intersection point
                        intersection = segment.intersection(road.geometry)
                        
                        # Create trimmed path segments
                        trimmed_segments.append(connecting_segment)
                        final_segment = shapely.geometry.LineString([
                            connecting_segment.coords[-1], 
                            intersection
                        ])
                        trimmed_segments.append(final_segment)
                        
                        road_intersection = True
                        intersected = True
                        break
                
                # If no intersection, add the connecting segment and current segment
                if not road_intersection:
                    trimmed_segments.append(connecting_segment)
                    trimmed_segments.append(segment)
                    last_point = segment.coords[-1]
                
                # Stop if we've found an intersection
                if intersected:
                    break
            
            # Combine segments into a single LineString
            if trimmed_segments:
                try:
                    combined_line = shapely.ops.linemerge(trimmed_segments)
                    
                    # Create a new GeoDataFrame row
                    trimmed_flow_path = flow_path.copy()
                    trimmed_flow_path['geometry'] = combined_line
                    return trimmed_flow_path
                except Exception as e:
                    print(f"Error merging segments: {e}")
                    return None
            
            return None

        # Process each flow path
        for _, flow_path in flow_paths.iterrows():
            trimmed_path = trim_single_flow_path(flow_path)
            
            if trimmed_path is not None:
                trimmed_flow_paths.append(trimmed_path)
        
        # Create a new GeoDataFrame with trimmed flow paths
        return gpd.GeoDataFrame(trimmed_flow_paths, crs=flow_paths.crs)

    # Main processing workflow
    # 1. Trace initial flow paths
    all_paths, flow_paths_gdf = trace_and_save_flow_paths(
        points_gdf=m.roads.drain_points_gdf,
        flow_proportions=flow_proportions,
        m=m
    )

    # 2. Trim flow paths based on road intersections
    trimmed_flow_paths_gdf = trim_flow_paths(
        flow_paths_gdf, 
        m.roads.drain_points_gdf, 
        m.roads.lines_gdf, 
        target_segment_length
    )

    # 3. Save trimmed flow paths
    trimmed_flow_paths_gdf.to_file('flow_paths_trimmed.shp')

    return trimmed_flow_paths_gdf

# Usage
result = process_flow_paths(m)

