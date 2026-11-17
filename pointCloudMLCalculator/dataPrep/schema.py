############################################################################################
"""
Input form definition for the UI (served at GET /schema).

Defaults come straight from the data check defaults so there's one source of truth.
Field types:
    number, integer, boolean, string, select     sent as params
    list_number, list_string                     sent as tables
show_if hides a field unless another field has one of the listed values.
"""
############################################################################################
try:
    from dataPrep.dataChecks import SYNTHETIC_DEFAULTS as D, GEOMETRIES, TARGET_FIELDS
except ImportError:
    from .dataChecks import SYNTHETIC_DEFAULTS as D, GEOMETRIES, TARGET_FIELDS

############################################################################################
def field(name='', label='', type='number', default=None, help='', options=None, min=None, max=None, show_if=None):
    f = {'name': name, 'label': label or name, 'type': type, 'default': default,
         'kind': 'table' if type.startswith('list_') else 'param'}
    if help: f['help'] = help
    if options is not None: f['options'] = options
    if min is not None: f['min'] = min
    if max is not None: f['max'] = max
    if show_if: f['show_if'] = show_if
    return f

############################################################################################
def getSchema():
    synthetic = {'dataset_mode': ['synthetic']}
    return {
        'name': 'pointcloud',
        'title': 'Point Cloud ML',
        'description': '3D point cloud workflows: anomaly detection, point-wise regression, segmentation and uncertainty.',
        'groups': [
            {'name': 'Run', 'fields': [
                field('name', 'Run name', 'string', 'PointCloud_Results'),
                field('problemType', 'Problem type', 'select', 'PROB01', options=['PROB01', 'PROB02', 'PROB03', 'PROB04'],
                      help='PROB01 anomaly detection, PROB02 regression, PROB03 segmentation, PROB04 uncertainty'),
                field('dataset_mode', 'Dataset', 'select', 'synthetic', options=['synthetic', 'existing']),
                field('csv_path', 'Point cloud CSV', 'string', '', show_if={'dataset_mode': ['existing']},
                      help='CSV with index, x, y, z, t, relative to the repo'),
                field('random_seed', 'Random seed', 'integer', D['random_seed'], show_if=synthetic),
            ]},
            {'name': 'Geometry', 'fields': [
                field('geometry', 'Shape', 'select', D['geometry'], options=GEOMETRIES, show_if=synthetic),
                field('n_points', 'Points', 'integer', D['n_points'], min=100, show_if=synthetic),
                field('size', 'Size', 'number', D['size'], min=0, show_if=synthetic),
                field('relief', 'Terrain relief', 'number', D['relief'], min=0, show_if={'dataset_mode': ['synthetic'], 'geometry': ['terrain']}),
                field('minor_radius_ratio', 'Torus tube ratio', 'number', D['minor_radius_ratio'], min=0.01, max=0.99,
                      show_if={'dataset_mode': ['synthetic'], 'geometry': ['torus']}),
                field('position_noise', 'Position noise', 'number', D['position_noise'], min=0, show_if=synthetic),
            ]},
            {'name': 'Target', 'fields': [
                field('target_field', 'Target field', 'select', D['target_field'], options=TARGET_FIELDS, show_if=synthetic),
                field('target_min', 'Target min', 'number', D['target_min'], show_if=synthetic),
                field('target_max', 'Target max', 'number', D['target_max'], show_if=synthetic),
                field('target_noise', 'Target noise', 'number', D['target_noise'], min=0, show_if=synthetic),
                field('n_field_centers', 'Field centers', 'integer', D['n_field_centers'], min=1,
                      show_if={'dataset_mode': ['synthetic'], 'target_field': ['smooth_field']}),
            ]},
            {'name': 'Anomalies', 'fields': [
                field('n_anomaly_patches', 'Anomaly patches', 'integer', D['n_anomaly_patches'], min=0, show_if=synthetic),
                field('anomaly_radius', 'Patch radius', 'number', D['anomaly_radius'], min=0, show_if=synthetic),
                field('anomaly_amplitude', 'Bump height', 'number', D['anomaly_amplitude'], min=0, show_if=synthetic),
                field('anomaly_target_shift', 'Target jump', 'number', D['anomaly_target_shift'], show_if=synthetic),
                field('outlier_fraction', 'Outlier fraction', 'number', D['outlier_fraction'], min=0, max=0.5, show_if=synthetic),
            ]},
        ],
    }
############################################################################################
