#!/usr/bin/env python3
"""
Data Format Validator for Spacecraft DR-MPC System
Validates data against standardized schemas and formats
"""

import json
import yaml
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormatValidator:
    """Comprehensive data format validation system."""
    
    def __init__(self, schemas_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.schemas_path = schemas_path or self.project_root / "docs" / "_data" / "data_formats.yml"
        self.schemas = self.load_schemas()
        
    def load_schemas(self) -> Dict:
        """Load data format schemas."""
        try:
            with open(self.schemas_path, 'r') as f:
                schemas = yaml.safe_load(f)
            logger.info(f"Loaded schemas from {self.schemas_path}")
            return schemas
        except FileNotFoundError:
            logger.error(f"Schema file {self.schemas_path} not found")
            return {}
    
    def validate_json_data(self, data: Dict, schema_name: str) -> tuple[bool, List[str]]:
        """Validate JSON data against schema."""
        if schema_name not in self.schemas:
            return False, [f"Schema {schema_name} not found"]
        
        schema = self.schemas[schema_name]
        errors = []
        
        try:
            validate(instance=data, schema=schema)
            return True, []
        except ValidationError as e:
            errors.append(f"Validation error: {e.message}")
            return False, errors
    
    def validate_spacecraft_state(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate spacecraft state data."""
        return self.validate_json_data(data, "spacecraft_state_schema")
    
    def validate_control_command(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate control command data."""
        return self.validate_json_data(data, "control_command_schema")
    
    def validate_test_result(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate test result data."""
        return self.validate_json_data(data, "test_result_schema")
    
    def validate_performance_metric(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate performance metric data."""
        return self.validate_json_data(data, "performance_metric_schema")
    
    def validate_csv_file(self, file_path: str, format_type: str) -> tuple[bool, List[str]]:
        """Validate CSV file format."""
        errors = []
        
        if format_type not in self.schemas.get("csv_format_specifications", {}):
            return False, [f"CSV format {format_type} not found in specifications"]
        
        spec = self.schemas["csv_format_specifications"][format_type]
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path, encoding=spec.get("encoding", "utf-8"))
            
            # Check required columns
            required_columns = [col["name"] for col in spec["columns"] if col.get("required", True)]
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            for col_spec in spec["columns"]:
                col_name = col_spec["name"]
                if col_name in df.columns:
                    col_type = col_spec["type"]
                    
                    if col_type == "datetime":
                        try:
                            pd.to_datetime(df[col_name])
                        except Exception as e:
                            errors.append(f"Invalid datetime format in column {col_name}: {e}")
                    
                    elif col_type == "numeric":
                        if not pd.api.types.is_numeric_dtype(df[col_name]):
                            errors.append(f"Column {col_name} should be numeric")
            
            # Check for empty rows
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                errors.append(f"Found {empty_rows} empty rows")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error reading CSV file: {e}"]
    
    def validate_yaml_config(self, file_path: str) -> tuple[bool, List[str]]:
        """Validate YAML configuration file."""
        errors = []
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                return False, ["Configuration must be a dictionary"]
            
            # Check required sections
            yaml_spec = self.schemas.get("yaml_configuration_format", {})
            required_sections = yaml_spec.get("required_sections", [])
            
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")
            
            # Validate section schemas
            section_schemas = yaml_spec.get("section_schemas", {})
            for section, schema in section_schemas.items():
                if section in config:
                    section_valid, section_errors = self.validate_section(config[section], schema)
                    if not section_valid:
                        errors.extend([f"{section}.{error}" for error in section_errors])
            
            return len(errors) == 0, errors
            
        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {e}"]
        except Exception as e:
            return False, [f"Error validating YAML: {e}"]
    
    def validate_section(self, data: Any, schema: Dict) -> tuple[bool, List[str]]:
        """Validate a configuration section against its schema."""
        errors = []
        
        if not isinstance(data, dict):
            return False, ["Section must be a dictionary"]
        
        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                errors.append(f"Missing required property: {prop}")
        
        # Validate properties
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in data:
                value = data[prop]
                
                # Type validation
                expected_type = prop_schema.get("type")
                if expected_type:
                    if not self.validate_type(value, expected_type):
                        errors.append(f"Property {prop} has invalid type")
                
                # Enum validation
                enum_values = prop_schema.get("enum")
                if enum_values and value not in enum_values:
                    errors.append(f"Property {prop} value {value} not in allowed values: {enum_values}")
                
                # Range validation
                if expected_type in ["number", "integer"]:
                    minimum = prop_schema.get("minimum")
                    maximum = prop_schema.get("maximum")
                    
                    if minimum is not None and value < minimum:
                        errors.append(f"Property {prop} value {value} below minimum {minimum}")
                    
                    if maximum is not None and value > maximum:
                        errors.append(f"Property {prop} value {value} above maximum {maximum}")
        
        return len(errors) == 0, errors
    
    def validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def validate_data_quality(self, data: Dict, data_type: str) -> tuple[bool, List[str]]:
        """Validate data quality requirements."""
        errors = []
        quality_reqs = self.schemas.get("data_quality_requirements", {})
        
        # Check completeness
        completeness_thresholds = quality_reqs.get("completeness", {}).get("thresholds", {})
        if data_type in completeness_thresholds:
            threshold = completeness_thresholds[data_type]
            completeness = self.calculate_completeness(data)
            if completeness < threshold:
                errors.append(f"Data completeness {completeness:.1f}% below threshold {threshold}%")
        
        # Check accuracy (if applicable)
        accuracy_reqs = quality_reqs.get("accuracy", {})
        if "position_data" in accuracy_reqs and "position" in data:
            tolerance = accuracy_reqs["position_data"]["tolerance"]
            if not self.validate_position_accuracy(data["position"], tolerance):
                errors.append(f"Position accuracy exceeds tolerance {tolerance}m")
        
        # Check freshness
        freshness_reqs = quality_reqs.get("freshness", {})
        if data_type in freshness_reqs:
            max_age = freshness_reqs[data_type]["max_age"]
            if "timestamp" in data:
                age = self.calculate_data_age(data["timestamp"])
                if age > max_age:
                    errors.append(f"Data age {age:.1f}s exceeds maximum {max_age}s")
        
        return len(errors) == 0, errors
    
    def calculate_completeness(self, data: Dict) -> float:
        """Calculate data completeness percentage."""
        if not isinstance(data, dict):
            return 0.0
        
        total_fields = 0
        complete_fields = 0
        
        for key, value in data.items():
            total_fields += 1
            if value is not None and value != "":
                complete_fields += 1
        
        return (complete_fields / total_fields) * 100.0 if total_fields > 0 else 0.0
    
    def validate_position_accuracy(self, position: List[float], tolerance: float) -> bool:
        """Validate position accuracy within tolerance."""
        if not isinstance(position, list) or len(position) != 3:
            return False
        
        # Simple validation - in practice, this would compare against reference
        return all(abs(p) < 10000 for p in position)  # Basic sanity check
    
    def calculate_data_age(self, timestamp: Union[str, float]) -> float:
        """Calculate data age in seconds."""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return (datetime.now().timestamp() - dt.timestamp())
            except ValueError:
                return float('inf')
        elif isinstance(timestamp, (int, float)):
            return datetime.now().timestamp() - timestamp
        else:
            return float('inf')
    
    def validate_file(self, file_path: str, format_type: str = None) -> tuple[bool, List[str]]:
        """Validate file based on extension or specified format."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, [f"File {file_path} does not exist"]
        
        # Determine format
        if format_type is None:
            format_type = self.detect_format(file_path)
        
        if format_type == "json":
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Try to determine schema based on content
                schema_name = self.detect_json_schema(data)
                if schema_name:
                    return self.validate_json_data(data, schema_name)
                else:
                    return True, []  # Valid JSON but unknown schema
                    
            except json.JSONDecodeError as e:
                return False, [f"Invalid JSON: {e}"]
        
        elif format_type == "csv":
            csv_format = self.detect_csv_format(file_path)
            if csv_format:
                return self.validate_csv_file(str(file_path), csv_format)
            else:
                return False, ["Could not determine CSV format"]
        
        elif format_type == "yaml":
            return self.validate_yaml_config(str(file_path))
        
        else:
            return False, [f"Unknown format type: {format_type}"]
    
    def detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        extension = file_path.suffix.lower()
        format_map = {
            ".json": "json",
            ".csv": "csv",
            ".yaml": "yaml",
            ".yml": "yaml"
        }
        return format_map.get(extension, "unknown")
    
    def detect_json_schema(self, data: Dict) -> Optional[str]:
        """Detect JSON schema based on content."""
        if "timestamp" in data and "position" in data and "velocity" in data:
            return "spacecraft_state_schema"
        elif "timestamp" in data and "forces" in data and "torques" in data:
            return "control_command_schema"
        elif "test_id" in data and "status" in data and "metrics" in data:
            return "test_result_schema"
        elif "metric_name" in data and "value" in data and "unit" in data:
            return "performance_metric_schema"
        else:
            return None
    
    def detect_csv_format(self, file_path: Path) -> Optional[str]:
        """Detect CSV format based on filename and content."""
        filename = file_path.stem.lower()
        
        if "performance" in filename:
            return "performance_data"
        elif "telemetry" in filename:
            return "telemetry_data"
        else:
            # Try to detect from column headers
            try:
                df = pd.read_csv(file_path, nrows=0)  # Read only headers
                columns = set(df.columns.str.lower())
                
                if {"timestamp", "metric_name", "value", "unit"}.issubset(columns):
                    return "performance_data"
                elif {"timestamp", "spacecraft_id", "position_x", "position_y", "position_z"}.issubset(columns):
                    return "telemetry_data"
            except Exception:
                pass
        
        return None
    
    def generate_validation_report(self, results: List[Dict]) -> str:
        """Generate comprehensive validation report."""
        total_files = len(results)
        valid_files = sum(1 for r in results if r["valid"])
        invalid_files = total_files - valid_files
        
        report = [
            "Data Format Validation Report",
            "=" * 40,
            f"Generated: {datetime.now().isoformat()}",
            f"Total files: {total_files}",
            f"Valid files: {valid_files}",
            f"Invalid files: {invalid_files}",
            f"Success rate: {(valid_files/total_files)*100:.1f}%" if total_files > 0 else "N/A",
            "",
            "Detailed Results:",
            "-" * 20
        ]
        
        for result in results:
            status = "✅ VALID" if result["valid"] else "❌ INVALID"
            report.append(f"{status} {result['file']}")
            
            if not result["valid"] and result["errors"]:
                for error in result["errors"]:
                    report.append(f"  • {error}")
                report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Data Format Validator for Spacecraft DR-MPC System")
    parser.add_argument("files", nargs="+", help="Files to validate")
    parser.add_argument("--format", choices=["json", "csv", "yaml"], help="Force specific format")
    parser.add_argument("--schemas", help="Path to schemas file")
    parser.add_argument("--report", help="Output report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = DataFormatValidator(args.schemas)
    results = []
    
    for file_path in args.files:
        logger.info(f"Validating {file_path}")
        valid, errors = validator.validate_file(file_path, args.format)
        
        result = {
            "file": file_path,
            "valid": valid,
            "errors": errors
        }
        results.append(result)
        
        if valid:
            print(f"✅ {file_path}: VALID")
        else:
            print(f"❌ {file_path}: INVALID")
            for error in errors:
                print(f"   • {error}")
    
    # Generate report
    report = validator.generate_validation_report(results)
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
    else:
        print("\n" + report)
    
    # Exit with appropriate code
    invalid_count = sum(1 for r in results if not r["valid"])
    sys.exit(0 if invalid_count == 0 else 1)

if __name__ == "__main__":
    main()