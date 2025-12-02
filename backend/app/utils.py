from app.supa import supa


def download_from_bucket(key: str, bucket: str) -> bytes:
    client = supa()
    return client.storage.from_(bucket).download(key)


def upload_to_bucket(key: str, data: bytes, bucket: str):
    client = supa()
    client.storage.from_(bucket).upload(
        path=key,
        file=data,
        file_options={
            "content-type": "text/csv",
            "upsert": "true",
        },
    )
