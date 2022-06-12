# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Plotting
import matplotlib.pyplot as plt

# Data science
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Geospatial
import contextily as cx
import xarray as xr
import zarr # Not referenced, but required for xarray

# Import Planetary Computer tools
import fsspec
import pystac

# Other
import os
import zipfile
from itertools import cycle

# MLFlow
import mlflow
import sys


# Path to data folder with provided material
data_path = 'data/'

if not os.path.exists(data_path+'training_data/'):
    os.mkdir(data_path+'training_data/')
    with zipfile.ZipFile(data_path+'GBIF_training_data.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path+'training_data/')
        
def filter_bbox(frogs, bbox):
    frogs = frogs[lambda x: 
        (x.decimalLongitude >= bbox[0]) &
        (x.decimalLatitude >= bbox[1]) &
        (x.decimalLongitude <= bbox[2]) &
        (x.decimalLatitude <= bbox[3])
    ]
    return frogs

def get_frogs(file, year_range=None, bbox=None):
    """Returns the dataframe of all frog occurrences for the bounding box specified."""
    columns = [
        'gbifID','eventDate','country','continent','stateProvince',
        'decimalLatitude','decimalLongitude','species', 'coordinateUncertaintyInMeters'
    ]
    country_names = {
        'AU':'Australia', 'CR':'Costa Rica', 'ZA':'South Africa','MX':'Mexico','HN':'Honduras',
        'MZ':'Mozambique','BW':'Botswana','MW':'Malawi','CO':'Colombia','PA':'Panama','NI':'Nicaragua',
        'BZ':'Belize','ZW':'Zimbabwe','SZ':'Eswatini','ZM':'Zambia','GT':'Guatemala','LS':'Lesotho',
        'SV':'El Salvador', 'AO':'Angola', np.nan:'unknown or invalid'
    }
    continent_names = {
        'AU':'Australia', 'CR':'Central America', 'ZA':'Africa','MX':'Central America','HN':'Central America',
        'MZ':'Africa','BW':'Africa','MW':'Africa','CO':'Central America','PA':'Central America',
        'NI':'Central America','BZ':'Central America','ZW':'Africa','SZ':'Africa','ZM':'Africa',
        'GT':'Central America','LS':'Africa','SV':'Central America','AO':'Africa', np.nan:'unknown or invalid' 
    }
    frogs = (
        pd.read_csv(data_path+'training_data/occurrence.txt', sep='\t', parse_dates=['eventDate'])
        .assign(
            country =  lambda x: x.countryCode.map(country_names),
            continent =  lambda x: x.countryCode.map(continent_names),
            species = lambda x: x.species.str.title()
        )
        [columns]
    )
    if year_range is not None:
        frogs = frogs[lambda x: 
            (x.eventDate.dt.year >= year_range[0]) & 
            (x.eventDate.dt.year <= year_range[1])
        ]
    if bbox is not None:
        frogs = filter_bbox(frogs, bbox)
    return frogs


def get_terraclimate(bbox, metrics, time_slice=None, assets=None, features=None, interp_dims=None, verbose=True):
    """Returns terraclimate metrics for a given area, allowing results to be interpolated onto a larger image.
    
    Attributes:
    bbox -- Tuple of (min_lon, min_lat, max_lon, max_lat) to define area
    metrics -- Nested dictionary in the form {<metric_name>:{'fn':<metric_function>,'params':<metric_kwargs_dict>}, ... }
    time_slice -- Tuple of datetime strings to select data between, e.g. ('2015-01-01','2019-12-31')
    assets -- list of terraclimate assets to take
    features -- list of asset metrics to take, specified by strings in the form '<asset_name>_<metric_name>'
    interp_dims -- Tuple of dimensions (n, m) to interpolate results to
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    collection = pystac.read_file("https://planetarycomputer.microsoft.com/api/stac/v1/collections/terraclimate")
    asset = collection.assets["zarr-https"]
    store = fsspec.get_mapper(asset.href)
    data = xr.open_zarr(store, **asset.extra_fields["xarray:open_kwargs"])
    
    # Select datapoints that overlap region
    if time_slice is not None:
        data = data.sel(lon=slice(min_lon,max_lon),lat=slice(max_lat,min_lat),time=slice(time_slice[0],time_slice[1]))
    else:
        data = data.sel(lon=slice(min_lon,max_lon),lat=slice(max_lat,min_lat))
    if assets is not None:
        data = data[assets]
    print('Loading data') if verbose else None
    data = data.rename(lat='y', lon='x').to_array().compute()
    
    print(f'Data Shape: {data.shape}')
        
    # Calculate metrics
    combined_values = []
    combined_bands = []
    for name, metric in metrics.items():
        print(f'Calculating {name}') if verbose else None
        sum_data = xr.apply_ufunc(
            metric['fn'], data, input_core_dims=[["time"]], kwargs=metric['params'], dask = 'allowed', vectorize = True
        ).rename(variable='band')
        
        xcoords = sum_data.x
        ycoords = sum_data.y
        dims = sum_data.dims
        print(f'Dimensions : {dims}')
        # print(f'Sum_data values {sum_data.values}')
        combined_values.append(sum_data.values)
        for band in sum_data.band.values:
            combined_bands.append(band+'_'+name)
    
    # Combine metrics
    combined_values = np.concatenate(
        combined_values,
        axis=0
    )
    combined_data = xr.DataArray(
        data=combined_values,
        dims=dims,
        coords=dict(
            band=combined_bands,
            y=ycoords,
            x=xcoords
        )
    )    

    # Take relevant bands:
    combined_data = combined_data.sel(band=features)
    
    if interp_dims is not None:
        print(f'Interpolating image') if verbose else None
        interp_coords = (np.linspace(bbox[0], bbox[2], interp_dims[0]), np.linspace(bbox[1], bbox[3], interp_dims[1]))
        combined_data = combined_data.interp(x=interp_coords[0], y=interp_coords[1], method='nearest', kwargs={"fill_value": "extrapolate"})
    
    return combined_data


def join_frogs(frogs, data):
    """Collects the data for each frog location and joins it onto the frog data 

    Arguments:
    frogs -- dataframe containing the response variable along with ["decimalLongitude", "decimalLatitude", "key"]
    data -- xarray dataarray of features, indexed with geocoordinates
    """
    return frogs.merge(
        (
            data
            .rename('data')
            .sel(
                x=xr.DataArray(all_frog_data.decimalLongitude, dims="key", coords={"key": all_frog_data.key}), 
                y=xr.DataArray(all_frog_data.decimalLatitude, dims="key", coords={"key": all_frog_data.key}),
                method="nearest"
            )
            .to_dataframe()
            .assign(val = lambda x: x.iloc[:, -1])
            [['val']]
            .reset_index()
            .drop_duplicates()
            .pivot(index="key", columns="band", values="val")
            .reset_index()
        ),
        on = ['key'],
        how = 'inner'
    )


def training(X, y, n_iter, cv):
    
    xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=123)

    params_random_search = {
        'learning_rate': np.arange(0.01, 1.01, 0.01),
        'n_estimators': np.arange(500, 2000),
        'max_depth': range(2, 5),
        'subsample': np.arange(0.02, 1.02, 0.02),
        'colsample_bytree': np.arange(0.3, 0.7, 0.1),
        'scale_pos_weight': np.arange(1, 3, 0.1)
    }

    randomized_cv = RandomizedSearchCV(estimator=xg_cl, param_distributions=params_random_search, scoring='roc_auc', n_iter=n_iter,
                cv=cv, verbose=2, n_jobs=-1)
    
    randomized_cv.fit(X, y)
    
    params = randomized_cv.best_params_
    
    xg_cl.set_params(**params)
    
    xg_cl.fit(X, y)
    
    validation_roc_auc = randomized_cv.cv_results_['mean_test_score'].mean()
    
    return xg_cl, params, validation_roc_auc


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Define the bounding box for Australia Region of Interest
    region_name = 'Greater Sydney, NSW'
    min_lon, min_lat = (115, -40.00)  # Lower-left corner
    max_lon, max_lat = (154.00, -10.00)  # Upper-right corner
    bbox = (min_lon, min_lat, max_lon, max_lat)

    bbox_range_list = [(144.8,-38.5,145.8,-37.5), (150.7,-33.5,151.7,-32.5), (152.6,-29.0,153.6,-28.0),
                (145.0,-17.7,146.0,-16.7), (115.7,-32.5,116.7,-31.5)]

    all_frog_data_dict = {}
    year_range_list = [(2014, 2015), (2016, 2017), (2018, 2019)]

    # Load in data
    for year_range in year_range_list:
        all_frog_data = get_frogs(data_path+'/training_data/occurrence.txt', year_range=year_range, bbox=bbox)
        all_frog_data_dict[year_range] = all_frog_data


    target_species = 'Litoria Fallax'

    for key, all_frog_data in all_frog_data_dict.items():
        all_frog_data = (
            all_frog_data
            # Assign the occurrenceStatus to 1 for the target species and 0 for all other species.
            # as well as a key for joining (later)
            .reset_index(drop = True)
            .assign(
                occurrenceStatus = lambda x: np.where(x.species == target_species, 1, 0),
                key = lambda x: x.index
            )
        )
        all_frog_data['coordinateUncertaintyInMeters'] = all_frog_data['coordinateUncertaintyInMeters'].fillna(0)
        all_frog_data = all_frog_data[all_frog_data['coordinateUncertaintyInMeters'] <= 100]
        
        all_frog_data_dict[key] = all_frog_data


    
    # Metrics to measure over time dimension
    tc_metrics = {
        'mean':{
            'fn':np.nanmean,
            'params':{}
        },
        'std':{
            'fn':np.nanstd,
            'params':{}
        },
        'median':{
            'fn':np.nanmedian,
            'params':{}
        },
        'min':{
            'fn':np.nanmax,
            'params':{}
        },
        'max':{
            'fn':np.nanmin,
            'params':{}
        }
    }

    # Date range to take
    time_slice = ('2014-01-01','2019-12-31')

    # Measurements to take
    assets = ['aet', 'def', 'pdsi', 'pet', 'ppt', 'ppt_station_influence', 'q', 'soil', 'srad', 'swe', 'tmax',
            'tmax_station_influence', 'tmin', 'tmin_station_influence', 'vap', 'vap_station_influence', 'vpd', 'ws']


    features = ['aet_mean', 'def_mean', 'pdsi_mean', 'pet_mean', 'ppt_mean', 'ppt_station_influence_mean', 'q_mean',
                'soil_mean', 'srad_mean', 'swe_mean', 'tmax_mean', 'tmax_station_influence_mean', 'tmin_mean',
                'tmin_station_influence_mean', 'vap_mean', 'vap_station_influence_mean', 'vpd_mean', 'ws_mean',
            'aet_std', 'def_std', 'pdsi_std', 'pet_std', 'ppt_std', 'ppt_station_influence_std', 'q_std',
                'soil_std', 'srad_std', 'swe_std', 'tmax_std', 'tmax_station_influence_std', 'tmin_std',
                'tmin_station_influence_std', 'vap_std', 'vap_station_influence_std', 'vpd_std', 'ws_std',
            'aet_min', 'def_min', 'pdsi_min', 'pet_min', 'ppt_min', 'ppt_station_influence_min', 'q_min', 'soil_min',
                'srad_min', 'swe_min', 'tmax_min', 'tmax_station_influence_min', 'tmin_min', 'tmin_station_influence_min',
                'vap_min', 'vap_station_influence_min', 'vpd_min', 'ws_min',
            'aet_max', 'def_max', 'pdsi_max', 'pet_max', 'ppt_max', 'ppt_station_influence_max', 'q_max', 'soil_max',
                'srad_max', 'swe_max', 'tmax_max', 'tmax_station_influence_max', 'tmin_max', 'tmin_station_influence_max',
                'vap_max', 'vap_station_influence_max', 'vpd_max', 'ws_max',
            'aet_median', 'def_median', 'pdsi_median', 'pet_median', 'ppt_median', 'ppt_station_influence_median', 'q_median',
                'soil_median', 'srad_median', 'swe_median', 'tmax_median', 'tmax_station_influence_median', 'tmin_median',
                'tmin_station_influence_median', 'vap_median', 'vap_station_influence_median', 'vpd_median', 'ws_median']

    weather_data_dict = {}
    time_slice_list = [('2014-01-01','2015-12-31'),
                    ('2016-01-01','2017-12-31'), ('2018-01-01','2019-12-31')]
    

    for time_slice in time_slice_list:
        weather_data = get_terraclimate(bbox, tc_metrics, time_slice=time_slice, assets=assets, features=features)
        #display(weather_data.band.values)
        
        weather_data_dict[time_slice] = weather_data



    all_model_data = []

    for all_frog_data, weather_data in zip(all_frog_data_dict.values(), weather_data_dict.values()):
        
        print(f'Frog data shape: {all_frog_data.shape}')
        print(f'Weather data shape: {weather_data.shape}')
        model_data = join_frogs(all_frog_data, weather_data)
        print(f'After merging shape: {model_data.shape}')
        
        all_model_data.append(model_data)



    df = all_model_data[0]

    for i in range(1, len(all_model_data)):
        df = pd.concat([df, all_model_data[i]], axis = 0)

    df = df[~df['ws_max'].isna()]
    model_data = df.copy()
    X = (
        model_data
        .drop(['gbifID', 'eventDate', 'decimalLatitude', 'decimalLongitude', 'species', 'coordinateUncertaintyInMeters',
        'stateProvince', 'country', 'continent', 'occurrenceStatus', 'key'], 1)
    )

    y = model_data.occurrenceStatus.astype(int)

    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cv = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print('\nTraining XGBoost model......')
    model, params, validation_roc_auc = training(X, y, n_iter, cv)
    
    mlflow.log_params(params)
    
    mlflow.log_metric('Validation ROC AUC', validation_roc_auc)
    
    mlflow.sklearn.log_model(model, 'XGBoost')
    