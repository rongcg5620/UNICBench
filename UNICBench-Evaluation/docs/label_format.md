# Dataset Label Format

This document describes the annotation format used in the Unified Counting Benchmark dataset.

## Overview

All modalities (image, text, audio) use a unified JSON annotation format with consistent structure and field definitions.

## JSON Structure

```json
{
    "type": "image|text|audio",
    "target_file_path": "relative_file_path_or_false",
    "attributes": {
        // Modality-specific metadata
    },
    "target_text": "text_content_for_text_modality",
    "questions": [
        {
            "question_id": 1,
            "question": "English question",
            "question_cn": "Chinese question", 
            "level": "Pattern|Semantic|Reasoning",
            "count": 42,
            "instances": [
                // Modality-specific instance annotations
            ]
        }
    ]
}
```

## Difficulty Levels

- **Pattern (L1)**: Direct perceptual counting of observable instances
- **Semantic (L2)**: Counting with attribute constraints and cross-segment deduplication  
- **Reasoning (L3)**: Rule-driven counting requiring multi-step reasoning

## Instance Formats by Modality

### Image
Point coordinates marking object centers:
```json
"instances": [
    [x_coordinate, y_coordinate],
    [x_coordinate, y_coordinate]
]
```

### Text  
Text spans with character positions:
```json
"instances": [
    {
        "text": "matched_text_fragment",
        "coordinates": {
            "start": 123,
            "end": 145
        }
    }
]
```

### Audio
Sound events with time ranges:
```json
"instances": [
    {
        "sound_type": "Speaker1_unknown",
        "time_range": [start_seconds, end_seconds]
    }
]
```

## Usage Notes

- All coordinates use 0-based indexing
- Image coordinates: origin at top-left corner
- Text coordinates: `[start, end)` character range
- Audio coordinates: time in seconds (float)
- `count` field matches `instances` array length
- Empty instances allowed when `count` is 0

This format ensures consistent evaluation across all modalities while preserving modality-specific annotation details.