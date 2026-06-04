import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Point


def bbox_to_geojson(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> dict:
    """
    Convert a bounding box to GeoJSON polygon format.
    """
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]
        ],
    }


def filter_above_boundary(
    df: pd.DataFrame,
    bounds_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Keep only points above a geographical boundary line.

    The boundary is given as a DataFrame with latitude and longitude columns.
    """
    boundary_line = LineString(zip(bounds_df[lon_col], bounds_df[lat_col]))

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
    )

    def is_above_line(point: Point, line: LineString) -> bool:
        projection = line.interpolate(line.project(point))
        return point.y > projection.y

    gdf["above"] = gdf.geometry.apply(lambda point: is_above_line(point, boundary_line))

    filtered = gdf[gdf["above"]].copy()

    return filtered.drop(columns=["geometry", "above"])


def project_to_nearest(
    coords_to_project: pd.DataFrame,
    coords_target: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    return_distance: bool = True,
) -> pd.DataFrame:
    """
    For each point in coords_to_project, find the nearest point in coords_target.

    Distances are computed using the haversine formula.
    """
    source = coords_to_project[[lat_col, lon_col]].to_numpy().astype(float)
    target = coords_target[[lat_col, lon_col]].to_numpy().astype(float)

    if source.size == 0 or target.size == 0:
        raise ValueError(
            "Empty inputs: both coords_to_project and coords_target must have rows."
        )

    try:
        from sklearn.neighbors import BallTree

        source_rad = np.radians(source)
        target_rad = np.radians(target)

        tree = BallTree(target_rad, metric="haversine")
        dist_rad, indices = tree.query(source_rad, k=1)

        nearest = target[indices[:, 0]]

        if return_distance:
            earth_radius_m = 6_371_000.0
            dist_m = dist_rad[:, 0] * earth_radius_m
        else:
            dist_m = None

    except Exception:
        source_exp = np.radians(source)[:, None, :]
        target_exp = np.radians(target)[None, :, :]

        dlat = target_exp[..., 0] - source_exp[..., 0]
        dlon = target_exp[..., 1] - source_exp[..., 1]

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(source_exp[..., 0])
            * np.cos(target_exp[..., 0])
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        earth_radius_m = 6_371_000.0
        distances = earth_radius_m * c

        indices = np.argmin(distances, axis=1)
        nearest = target[indices]

        if return_distance:
            dist_m = distances[np.arange(distances.shape[0]), indices]
        else:
            dist_m = None

    output = pd.DataFrame(
        {
            lat_col: source[:, 0],
            lon_col: source[:, 1],
            "lat_nearest": nearest[:, 0],
            "lon_nearest": nearest[:, 1],
        }
    )

    if return_distance:
        output["distance_m"] = dist_m

    return output
