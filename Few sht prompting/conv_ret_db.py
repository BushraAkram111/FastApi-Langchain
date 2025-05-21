# SQLAlchemy imports to define tables and connect to PostgreSQL
from sqlalchemy import create_engine, Column, String, Text, BigInteger
# Declarative base is used to define ORM models (tables)
from sqlalchemy.orm import declarative_base, sessionmaker

# For accessing environment variables
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base class used for creating table definitions (ORM)
Base = declarative_base()

# 🔷 Define the database table model using SQLAlchemy ORM
class ConversationChatHistory(Base):
    __tablename__ = 'conversation_chain'  # Table name in the database

    # Auto-incrementing primary key column
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Chatbot session/user identifier
    chatbot_id = Column(String, nullable=False)

    # User's question
    query = Column(Text, nullable=False)

    # AI model's response
    response = Column(Text, nullable=False)

# 🔷 Build the PostgreSQL connection URL using environment variables
DATABASE_URL = f"postgresql://{os.getenv('PG_USER_NAME')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_NAME')}"

# 🔧 Create a database engine that connects to PostgreSQL
engine = create_engine(DATABASE_URL)

# 🔄 Create a session class for interacting with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Automatically create the table in the database if it doesn't exist
Base.metadata.create_all(bind=engine)
