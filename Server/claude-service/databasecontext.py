from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import json
from datetime import datetime

@dataclass
class ColumnInfo:
    """Detailed column information"""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    unique: bool = False
    default_value: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)

@dataclass
class TableInfo:
    """Enhanced table information"""
    name: str
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    row_count: int = 0
    sample_data: List[Dict[str, Any]] = field(default_factory=list)
    relationships: Set[str] = field(default_factory=set)

@dataclass
class DatabaseContext:
    """Enhanced database context with rich metadata"""
    schema_text: str
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    rag_collection_name: Optional[str] = None
    last_analyzed: Optional[datetime] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def get_table_names(self) -> List[str]:
        """Get all table names"""
        return list(self.tables.keys())
    
    def get_column_names(self, table_name: str) -> List[str]:
        """Get column names for a specific table"""
        if table_name in self.tables:
            return list(self.tables[table_name].columns.keys())
        return []
    
    def get_all_columns(self) -> Dict[str, List[str]]:
        """Get all columns organized by table"""
        return {table_name: list(table_info.columns.keys()) 
                for table_name, table_info in self.tables.items()}
    
    def find_relationship_path(self, from_table: str, to_table: str) -> Optional[List[str]]:
        """Find the shortest path between two tables via foreign keys"""
        if from_table == to_table:
            return [from_table]
        
        visited = set()
        queue = [(from_table, [from_table])]
        
        while queue:
            current_table, path = queue.pop(0)
            if current_table in visited:
                continue
                
            visited.add(current_table)
            
            if current_table in self.relationships:
                for related_table in self.relationships[current_table]:
                    if related_table == to_table:
                        return path + [related_table]
                    if related_table not in visited:
                        queue.append((related_table, path + [related_table]))
        
        return None
    
    def get_joinable_tables(self, table_name: str) -> List[str]:
        """Get tables that can be joined with the given table"""
        return self.relationships.get(table_name, [])
    
    def get_table_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_tables": len(self.tables),
            "total_columns": sum(len(table.columns) for table in self.tables.values()),
            "total_rows": sum(table.row_count for table in self.tables.values()),
            "tables_with_fk": len([t for t in self.tables.values() if t.foreign_keys]),
            "last_analyzed": self.last_analyzed.isoformat() if self.last_analyzed else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "schema_text": self.schema_text,
            "tables": {name: {
                "name": table.name,
                "columns": {col_name: {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "foreign_key": col.foreign_key,
                    "unique": col.unique,
                    "default_value": col.default_value,
                    "sample_values": col.sample_values
                } for col_name, col in table.columns.items()},
                "foreign_keys": table.foreign_keys,
                "row_count": table.row_count,
                "sample_data": table.sample_data,
                "relationships": list(table.relationships)
            } for name, table in self.tables.items()},
            "relationships": self.relationships,
            "rag_collection_name": self.rag_collection_name,
            "last_analyzed": self.last_analyzed.isoformat() if self.last_analyzed else None,
            "statistics": self.statistics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseContext':
        """Deserialize from dictionary"""
        context = cls(schema_text=data["schema_text"])
        
        # Reconstruct tables
        for table_name, table_data in data.get("tables", {}).items():
            table_info = TableInfo(name=table_data["name"])
            
            # Reconstruct columns
            for col_name, col_data in table_data.get("columns", {}).items():
                table_info.columns[col_name] = ColumnInfo(
                    name=col_data["name"],
                    type=col_data["type"],
                    nullable=col_data["nullable"],
                    primary_key=col_data["primary_key"],
                    foreign_key=col_data.get("foreign_key"),
                    unique=col_data.get("unique", False),
                    default_value=col_data.get("default_value"),
                    sample_values=col_data.get("sample_values", [])
                )
            
            table_info.foreign_keys = table_data.get("foreign_keys", [])
            table_info.row_count = table_data.get("row_count", 0)
            table_info.sample_data = table_data.get("sample_data", [])
            table_info.relationships = set(table_data.get("relationships", []))
            
            context.tables[table_name] = table_info
        
        context.relationships = data.get("relationships", {})
        context.rag_collection_name = data.get("rag_collection_name")
        context.statistics = data.get("statistics", {})
        
        if data.get("last_analyzed"):
            context.last_analyzed = datetime.fromisoformat(data["last_analyzed"])
        
        return context