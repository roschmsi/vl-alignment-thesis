import os

SCRATCH = "/dss/mcmlscratch/07/ga27tus3"

DATADIR = {
    "dreamclipcc3m": {
        "annotation": f"{SCRATCH}/cc3m_3long_3short_1raw_captions_url_filtered.csv",
        "imagedir": f"{SCRATCH}",
    },
    # 'dreamclipcc12m': {
    #     'annotation':f'{SCRATCH}/datasets/DownloadCC3M/cc12m_3long_3short_1raw_captions_url_filtered.csv',
    #     'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
    #     },
    # 'yfcc15m': {
    #     'annotation':f'{SCRATCH}/datasets/DownloadCC3M/yfcc15m_3long_3short_1raw_captions_url_filtered.csv',
    #     'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
    #     },
}
