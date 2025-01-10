from sqlalchemy import Table, Column, ForeignKey
from sqlalchemy.sql.sqltypes import Integer, String
from config.database import meta

users = Table(
    'users', meta, 
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(255), nullable=False),
    Column('email', String(255), nullable=False, unique=True),
    Column('password', String(255), nullable=False)
)

addresses = Table(
    'addresses', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
    Column('street', String(255), nullable=False),
    Column('zipcode', String(20), nullable=False),
    Column('country', String(100), nullable=False)
)

