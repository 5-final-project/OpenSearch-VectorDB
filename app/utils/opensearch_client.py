from opensearchpy import OpenSearch
from app.config.settings import Settings

settings = Settings()
client = OpenSearch(hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}])