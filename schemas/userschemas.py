from pydantic import BaseModel

class UserBase(BaseModel):
    name: str
    username: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True

class AddressBase(BaseModel):
    street: str
    zipcode: str
    country: str

class AddressCreate(AddressBase):
    pass

class Address(AddressBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True
