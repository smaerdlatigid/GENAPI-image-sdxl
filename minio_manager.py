import os
import json
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin

import numpy as np
from PIL import Image
from minio import Minio
from minio.error import S3Error

class MinioStorageManager:
    def __init__(
        self, 
        endpoint: str = "localhost:9000",
        access_key: str = "ROOTUSER",
        secret_key: str = "ROOTPASSWORD",
        bucket: str = "test",
        secure: bool = False,
        external_endpoint: str = None  # Optional external endpoint for public URLs
    ):
        self.endpoint = endpoint
        self.external_endpoint = external_endpoint or endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure

        # Initialize MinIO client
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def list_buckets(self) -> List[Dict]:
        """List all buckets"""
        try:
            buckets = self.client.list_buckets()
            return [{"name": bucket.name, "creation_date": bucket.creation_date} for bucket in buckets]
        except S3Error as e:
            print(f"Error listing buckets: {e}")
            return []

    def files(self, path_on_storage: str = "") -> List[Dict]:
        """List all files in a bucket with optional prefix"""
        try:
            objects = self.client.list_objects(self.bucket, prefix=path_on_storage, recursive=True)
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "url": self.get_file_url(obj.object_name)
                }
                for obj in objects
            ]
        except S3Error as e:
            print(f"Error listing files: {e}")
            return []

    def create_bucket(self, bucket: str = None, public: bool = False) -> Dict:
        """Create a new bucket"""
        if not bucket:
            bucket = self.bucket
        
        try:
            # Check if bucket already exists
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                
                # If public access is requested, set bucket policy
                if public:
                    policy = {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"AWS": "*"},
                                "Action": ["s3:GetObject"],
                                "Resource": [f"arn:aws:s3:::{bucket}/*"]
                            }
                        ]
                    }
                    self.client.set_bucket_policy(bucket, json.dumps(policy))
                
                return {"name": bucket, "status": "created"}
            return {"name": bucket, "status": "already exists"}
            
        except S3Error as e:
            print(f"Error creating bucket: {e}")
            return {"name": bucket, "status": "error", "message": str(e)}
    
    def upload_file_from_stream(
        self,
        file_stream,
        object_name: str,
        content_type: str,
        bucket: str
    ):
        # Get the size of the file stream
        file_size = file_stream.seek(0, os.SEEK_END)
        file_stream.seek(0)

        self.client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=file_stream,
            length=file_size,
            content_type=content_type
        )
    
    def get_file_url(self, object_name: str, bucket: str = None, expires: int = None) -> str:
        """
        Get URL for a file. If expires is set, returns a presigned URL.
        Otherwise returns a public URL.
        """
        if not bucket:
            bucket = self.bucket

        try:
            if expires:
                # Get presigned URL for temporary access
                return self.client.presigned_get_object(
                    bucket,
                    object_name,
                    expires=expires
                )
            else:
                # Ensure the external endpoint is set and properly formatted
                if not self.external_endpoint:
                    raise ValueError("External endpoint is not configured.")
                
                # Construct public URL
                # Use urllib.parse.urljoin to ensure correct URL formatting
                
                base_url = f"{self.external_endpoint}/{bucket}/"
                file_url = urljoin(base_url, object_name)
                
                return file_url

        except S3Error as e:
            print(f"MinIO error generating URL: {e}")
            return None
        except ValueError as ve:
            print(f"Configuration error: {ve}")
            return None
        except Exception as ex:
            print(f"Unexpected error generating URL: {ex}")
            return None

    def upload_file(
        self, 
        filepath: Path, 
        path_on_storage: str, 
        content_type: str = None,
        bucket: str = None,
        make_public: bool = True
    ) -> str:
        """Upload a file to storage"""
        if not bucket:
            bucket = self.bucket

        try:
            # Ensure bucket exists
            if not self.client.bucket_exists(bucket):
                self.create_bucket(bucket)

            # If content_type is not provided, try to guess it
            if not content_type:
                content_type, _ = mimetypes.guess_type(str(filepath))

            # Upload the file
            self.client.fput_object(
                bucket,
                path_on_storage,
                str(filepath),
                content_type=content_type
            )

            # Return the appropriate URL
            if make_public:
                return self.get_file_url(path_on_storage, bucket)
            else:
                # Return a presigned URL that expires in 7 days
                return self.get_file_url(path_on_storage, bucket, expires=7*24*60*60)

        except S3Error as e:
            print(f"Error uploading file: {e}")
            return None

    # download: response = self.client.get_object(bucket, path_on_storage); response.data

    def download_file_to_disk(self, path_on_storage: str, local_path: Path, bucket: str = None) -> None:
        if not bucket:
            bucket = self.bucket

        try:
            self.client.fget_object(bucket, path_on_storage, str(local_path))
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False

    @staticmethod
    def hash_file(file_path: Path) -> str:
        sha3_hash = hashlib.sha3_256()

        try:
            with file_path.open('rb') as f:
                while chunk := f.read(8192):
                    sha3_hash.update(chunk)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        return sha3_hash.hexdigest()

    @staticmethod
    def resize_image(img_path, target_size=(200, 100)):
        """Resizes an image to a thumbnail while maintaining the aspect ratio"""
        img = Image.open(str(img_path))
        w, h = img.size
        scale_factor = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        resized_img = img.resize(new_size, Image.BILINEAR)
        return np.array(resized_img)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MinIO Storage Manager')
    parser.add_argument('--create-bucket', type=str, help='Create a new bucket')
    parser.add_argument('--bucket', type=str, help='Default bucket for manager', default='test')
    parser.add_argument('--upload', type=str, help='Upload a file')
    parser.add_argument('--list', action='store_true', help='List files in bucket')
    args = parser.parse_args()

    # load from .env file
    from dotenv import load_dotenv
    load_dotenv()

    manager = MinioStorageManager(
        endpoint=os.environ['MINIO_ENDPOINT'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_KEY'],
        external_endpoint=os.environ['MINIO_EXTERNAL_ENDPOINT'],
        bucket=args.bucket,
    )

    if args.create_bucket:
        manager.bucket = args.create_bucket
        response = manager.create_bucket(public=True)
        print(f"Bucket {manager.bucket} created: {response}")

    if args.list:
        files = manager.files()
        print("Files in bucket:")
        for file in files:
            print(f"- {file['name']} ({file['size']} bytes) - {file['url']}")

    if args.upload:
        file_path = Path(args.upload)
        if file_path.exists():
            url = manager.upload_file(
                file_path,
                file_path.name,
                make_public=True
            )
            print(f"Uploaded file URL: {url}")
        else:
            print(f"File not found: {args.upload}")

    # List all buckets
    buckets = manager.list_buckets()
    print(f"All buckets: {buckets}")
