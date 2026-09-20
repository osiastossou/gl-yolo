from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Liste tous les fichiers générés par le kernel v13
files = api.kernels_output_cli('osiastossou/gl-yolo/13')

# Télécharger uniquement les fichiers du dossier voulu
prefix = "runs/detect/yolo11-ccp-200/"
for file_info in files:
    file_path = file_info.string_file_path
    if file_path.startswith(prefix):
        api.kernel_output_file_download(
            'osiastossou/gl-yolo/13',
            file_path,
            path='/Users/osias/Downloads/yolo11-ccp-200'
        )