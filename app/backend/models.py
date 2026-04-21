from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ChatStatus(str, Enum):
    SETUP = "setup"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ExtractionMethod(str, Enum):
    TEXT = "text_extraction"
    VISION_TABLE = "vision_table_parse"
    VISION_OCR = "vision_ocr"
    VISION_FIGURE = "vision_figure"


class ContentNode(BaseModel):
    type: str  # heading, paragraph, list, table, figure
    content: Any
    page: int
    extraction_method: ExtractionMethod = ExtractionMethod.TEXT
    confidence: float = 1.0


class DocumentSection(BaseModel):
    heading: Optional[str] = None
    page: int
    content: str
    content_nodes: List[ContentNode] = []
    subsections: List["DocumentSection"] = []


class ExtractedDocument(BaseModel):
    source: str
    pages: int
    sections: List[DocumentSection]
    pages_detail: List[Dict[str, Any]] = []


class Entity(BaseModel):
    id: str
    name: str
    type: str  # Person, Organization, Concept, Claim, Date, Metric, Policy, Event, Location
    description: Optional[str] = None
    source_doc: str
    source_page: Optional[int] = None
    source_section: Optional[str] = None


class Relationship(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    description: Optional[str] = None
    source_doc: str
    source_page: Optional[int] = None
    confidence: float = 1.0


class DocumentAnalysis(BaseModel):
    document_source: str
    summary: str
    key_claims: List[str] = []
    entities: List[Entity] = []
    relationships: List[Relationship] = []
    cross_doc_signals: List[str] = []


class CrossDocumentAnalysis(BaseModel):
    shared_entities: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []
    complementary_connections: List[Dict[str, Any]] = []
    temporal_relationships: List[Dict[str, Any]] = []


# ── API request / response models ────────────────────────────────────────────

class APIKeyRequest(BaseModel):
    openai_api_key: str


class APIKeyResponse(BaseModel):
    has_key: bool
    message: str


class CreateChatResponse(BaseModel):
    chat_id: str
    message: str


class ProcessingStatus(BaseModel):
    chat_id: str
    status: ChatStatus
    stage: str
    progress: float  # 0-100
    details: Optional[str] = None


class ChatListItem(BaseModel):
    chat_id: str
    title: str
    status: ChatStatus
    created_at: datetime
    updated_at: datetime
    document_count: int


class ChatDetail(BaseModel):
    chat_id: str
    title: str
    status: ChatStatus
    created_at: datetime
    messages: List[Dict[str, Any]]
    document_count: int


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: str
    retrieval_method: str  # "graph", "rlm", "hybrid"
    sources: List[Dict[str, Any]] = []
    graph_insights: Optional[str] = None


class GraphSummary(BaseModel):
    chat_id: str
    entity_count: int
    relationship_count: int
    document_count: int
    top_entities: List[Dict[str, Any]] = []
    top_relationships: List[str] = []
