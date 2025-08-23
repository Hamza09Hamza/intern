from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DatabaseContext:
    schema_text: str
    tables: List[str]
    columns_by_table: Dict[str, List[str]]
    rag_collection_name: Optional[str] = None