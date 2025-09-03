from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# The entity types you will support
EntityType = Literal[
    "OBJECT", "COLLECTION", "DATASET", "PROJECT", "SPACE",  
    "PROPERTY_TYPE", "SAMPLE_TYPE", "PROJECT_TYPE", "VOCABULARY" 
]

# The actions the agent can take
ActionType = Literal["GET_ONE", "LIST", "GET_TYPE"]

class Criteria(BaseModel):
    """Parameters for finding entities."""
    identifier: Optional[str] = Field(None, description="A unique identifier like /SPACE/PROJECT/CODE or a permId.")
    space: Optional[str] = Field(None, description="The space to search within.")
    project: Optional[str] = Field(None, description="The project to search within.")
    collection: Optional[str] = Field(None, description="The collection/experiment to search within.")
    type: Optional[str] = Field(None, description="The entity type code (e.g., 'YEAST').")
    where: Optional[Dict[str, Any]] = Field(None, description="A dictionary for advanced property/attribute filters, e.g., {'registrationDate': '>2024-01-01'}.")

class SortInstruction(BaseModel):
    """Defines how to sort by a single field."""
    field: str = Field(..., description="The field to sort by, e.g., 'registrationDate' or 'code'.")
    order: Literal["asc", "desc"] = Field("desc", description="The sort order, ascending or descending.")

class FetchOptions(BaseModel):
    """Parameters that control the data returned by the query."""
    properties: Optional[List[str]] = Field(None, description="Specific properties to return. Use ['*'] for all properties.")
    attributes: Optional[List[str]] = Field(None, description="Specific attributes to return (e.g., ['parents', 'children']).")
    limit: Optional[int] = Field(None, description="The maximum number of results to return.")
    sort_by: Optional[List[SortInstruction]] = Field(None, description="A list of instructions on how to sort the results.")

class ActionItem(BaseModel):
    """A single, complete action to be performed."""
    action: ActionType = Field(..., description="The type of action to perform.")
    entity: EntityType = Field(..., description="The type of entity the action applies to.")
    criteria: Optional[Criteria] = Field({}, description="The criteria for finding the entity/entities.")
    fetch_options: Optional[FetchOptions] = Field({}, description="Options for what data to return.")
    # payload: Optional[Dict[str, Any]] = Field({}, description="Data for CREATE/UPDATE actions. To be used in the future.")

class ActionRequest(BaseModel):
    """The root model representing a complete request from the user, potentially containing multiple actions."""
    actions: List[ActionItem] = Field(..., description="A list of actions to be executed based on the user's query.")
