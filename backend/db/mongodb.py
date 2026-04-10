from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

db_manager = MongoDB()

async def connect_to_mongo():
    logger.info("Connecting to MongoDB...")
    db_manager.client = AsyncIOMotorClient(settings.MONGO_URI)
    db_manager.db = db_manager.client[settings.DB_NAME]
    logger.info("Connected to MongoDB!")

async def close_mongo_connection():
    logger.info("Closing MongoDB connection...")
    if db_manager.client:
        db_manager.client.close()
    logger.info("MongoDB connection closed.")

def get_db():
    return db_manager.db
