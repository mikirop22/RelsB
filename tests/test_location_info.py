import osmnx as ox
import pandas as pd
from shapely.geometry import Point, box
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def download_illinois_data():
    """Download and cache Illinois OSM data"""
    # Define Illinois bounds approximately
    illinois_bounds = (36.9701, 42.5084, -91.513, -87.0199)  # (south, north, west, east)
    
    # Download and cache data for the whole state
    tags = {
        'leisure': ['park'],
        'amenity': ['school', 'restaurant'],
        'shop': True,
        'public_transport': True
    }
    
    # This will automatically cache the data locally
    G = ox.graph_from_bbox(illinois_bounds[1], illinois_bounds[0], 
                          illinois_bounds[3], illinois_bounds[2], 
                          network_type=None, 
                          custom_filter=f'[out:json];(way{ox.settings.default_overpass_query_settings});(._;>;);out;')
    
    return G

def get_amenity_counts_batch(df, radius=1000, batch_size=1000):
    """
    Process amenity counts for entire dataframe efficiently using cached data
    
    Args:
        df: DataFrame with 'Location.GIS.Latitude' and 'Location.GIS.Longitude' columns
        radius: Search radius in meters
        batch_size: Number of locations to process in parallel
    """
    # Configure OSMnx to use cache
    ox.config(use_cache=True, log_console=False)
    
    def process_single_location(row):
        lat, lon = row['Location.GIS.Longitude'], row['Location.GIS.Latitude']
        
        # Create bounding box for the location
        delta = radius / 111000  # Convert meters to approximate degrees
        bbox = box(lon - delta, lat - delta, lon + delta, lat + delta)
        
        # Get amenities within the bbox from cached data
        tags = {
            'leisure': ['park'],
            'amenity': ['school', 'restaurant'],
            'shop': True,
            'public_transport': True
        }
        
        try:
            amenities = ox.geometries_from_bbox(
                bbox.bounds[3], bbox.bounds[1], 
                bbox.bounds[2], bbox.bounds[0],
                tags
            )
            
            # Count amenities
            counts = {
                'parks': len(amenities[amenities['leisure'] == 'park']),
                'schools': len(amenities[amenities['amenity'] == 'school']),
                'restaurants': len(amenities[amenities['amenity'] == 'restaurant']),
                'shops': len(amenities[amenities['shop'].notna()]),
                'transport': len(amenities[amenities['public_transport'].notna()])
            }
            
        except Exception:
            counts = {
                'parks': 0, 'schools': 0, 'restaurants': 0,
                'shops': 0, 'transport': 0
            }
        
        return counts
    
    # Process in parallel using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=min(batch_size, 32)) as executor:
        futures = []
        for _, row in df.iterrows():
            futures.append(executor.submit(process_single_location, row))
        
        results = [future.result() for future in futures]
    
    # Convert results to DataFrame columns
    for amenity in ['parks', 'schools', 'restaurants', 'shops', 'transport']:
        df[f'count_{amenity}'] = [result[amenity] for result in results]
    
    return df

# Example usage
if __name__ == "__main__":
    # Example dataframe
    df = pd.read_csv("./data/train.csv")
    
    # Process the dataframe
    df = get_amenity_counts_batch(df)
    df.to_csv("./results.csv")
    print(df)