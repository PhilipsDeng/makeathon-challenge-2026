import json
from collections import Counter

def check_geojson(filepath):
    results = {}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        results['is_json'] = True
    except Exception as e:
        results['is_json'] = False
        results['error'] = str(e)
        return results

    results['type_FeatureCollection'] = data.get('type') == 'FeatureCollection'
    features = data.get('features')
    results['features_is_list'] = isinstance(features, list)
    
    if results['features_is_list']:
        results['num_features'] = len(features)
        types = Counter()
        geom_types = Counter()
        has_geom_prop = True
        coord_dims = Counter()
        prop_keys = set()
        time_step_types = Counter()
        lat_range = [float('inf'), float('-inf')]
        lon_range = [float('inf'), float('-inf')]
        empty_geom_count = 0
        
        for feat in features:
            types[feat.get('type')] += 1
            geom = feat.get('geometry')
            prop = feat.get('properties')
            
            if geom is None or prop is None:
                has_geom_prop = False
            
            if geom:
                g_type = geom.get('type')
                geom_types[g_type] += 1
                coords = geom.get('coordinates')
                if coords:
                    # Simple dimension check for Points/Polygons (first element)
                    def check_dims(c):
                        if isinstance(c[0], (int, float)):
                            return len(c)
                        return check_dims(c[0])
                    
                    try:
                        d = check_dims(coords)
                        coord_dims[d] += 1
                        
                        # Range check
                        def traverse_coords(c):
                            if isinstance(c[0], (int, float)):
                                lon, lat = c[0], c[1]
                                lon_range[0] = min(lon_range[0], lon)
                                lon_range[1] = max(lon_range[1], lon)
                                lat_range[0] = min(lat_range[0], lat)
                                lat_range[1] = max(lat_range[1], lat)
                            else:
                                for sub in c: traverse_coords(sub)
                        traverse_coords(coords)
                    except:
                        pass
                else:
                    empty_geom_count += 1
            else:
                empty_geom_count += 1
                
            if prop is not None:
                prop_keys.update(prop.keys())
                ts = prop.get('time_step')
                time_step_types[type(ts).__name__] += 1
        
        results['feature_types'] = dict(types)
        results['geometry_types'] = dict(geom_types)
        results['has_geom_prop'] = has_geom_prop
        results['coord_dims'] = dict(coord_dims)
        results['prop_keys'] = prop_keys
        results['time_step_types'] = dict(time_step_types)
        results['lon_range'] = lon_range
        results['lat_range'] = lat_range
        results['empty_geom_count'] = empty_geom_count
        
    return results

res4 = check_geojson('submission/submission-4.geojson')
res_new = check_geojson('submission/submission.geojson')

def print_res(name, res):
    print(f"--- {name} ---")
    if not res['is_json']:
        print("JSON invalid")
        return
    print(f"FC: {res['type_FeatureCollection']}, List: {res['features_is_list']}, Count: {res.get('num_features')}")
    print(f"GeomTypes: {res.get('geometry_types')}")
    print(f"Dims: {res.get('coord_dims')}")
    print(f"TS Types: {res.get('time_step_types')}")
    print(f"Lon Range: {res.get('lon_range')}")
    print(f"Lat Range: {res.get('lat_range')}")
    print(f"Empty: {res.get('empty_geom_count')}")
    print(f"Prop Keys: {res.get('prop_keys')}")

print_res("submission-4.geojson", res4)
print_res("submission.geojson", res_new)

if res4['is_json'] and res_new['is_json']:
    diff_keys = res4['prop_keys'] ^ res_new['prop_keys']
    print(f"\nDiff Keys: {diff_keys}")
    consistent = (res4['type_FeatureCollection'] == res_new['type_FeatureCollection'] and 
                  res4['features_is_list'] == res_new['features_is_list'] and
                  res4['prop_keys'] == res_new['prop_keys'] and
                  res4['geometry_types'].keys() == res_new['geometry_types'].keys())
    print(f"Consistent: {consistent}")
