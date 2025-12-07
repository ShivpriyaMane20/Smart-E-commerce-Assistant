# backend/api.py
# UPDATED WITH INTEGRATED SECURITY GUARDRAILS

import io
import re
from typing import Any, Dict, List
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

from .core import (
    full_analysis,
    generate_reviews_for_product,
    build_report_text,
    build_report_html,
    build_report_pdf,
    generate_improved_description,
    generate_caption_from_suggestions,
)


app = FastAPI(title="Smart E-Commerce Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SECURITY: INTEGRATED PROMPT INJECTION DETECTION
# ============================================================================

def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Detect prompt injection attempts in user input.
    Returns: (is_injection, reason)
    """
    if not text or not isinstance(text, str):
        return False, ""
    
    text_lower = text.lower().strip()
    
    # Pattern 1: Role manipulation
    role_patterns = [
        r"you\s+are\s+(no\s+longer|now|actually|really)\s+(a|an)\s+\w+",
        r"act\s+as\s+(a|an)\s+\w+",
        r"pretend\s+(to\s+be|you\s+are)",
    ]
    for pattern in role_patterns:
        if re.search(pattern, text_lower):
            return True, "Role manipulation detected"
    
    # Pattern 2: Instruction override
    override_patterns = [
        r"ignore\s+(all\s+)?(previous|above|prior|your)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|above|prior)",
        r"forget\s+(all\s+)?(previous|above|prior)",
        r"new\s+(instructions?|task|role|prompt)",
    ]
    for pattern in override_patterns:
        if re.search(pattern, text_lower):
            return True, "Instruction override attempt detected"
    
    # Pattern 3: System prompt extraction
    extraction_patterns = [
        r"(show|reveal|display|tell|give)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
        r"repeat\s+(your|the)\s+(system\s+)?(prompt|instructions?)",
        r"what\s+(are|were|is)\s+(your|the)\s+(original\s+)?(instructions?|prompt)",
    ]
    for pattern in extraction_patterns:
        if re.search(pattern, text_lower):
            return True, "System prompt extraction attempt detected"
    
    # Pattern 4: Out-of-scope requests
    outofscope_patterns = [
        (r"(financial|investment|stock|crypto|trading)\s+(advice|tips|recommendations?)", 
         "Out-of-scope: Financial advice"),
        (r"(medical|health|diagnosis|treatment)\s+(advice|tips)", 
         "Out-of-scope: Medical advice"),
        (r"(legal|lawyer|attorney)\s+(advice|counsel)", 
         "Out-of-scope: Legal advice"),
    ]
    for pattern, reason in outofscope_patterns:
        if re.search(pattern, text_lower):
            return True, reason
    
    # Pattern 5: Jailbreak attempts
    jailbreak_keywords = ["dan", "jailbreak", "bypass", "unrestricted", "sudo mode", "god mode"]
    if any(keyword in text_lower for keyword in jailbreak_keywords):
        return True, "Jailbreak attempt detected"
    
    # Pattern 6: Multiple suspicious keywords
    suspicious = ["ignore", "override", "bypass", "hack", "exploit", "unrestricted"]
    count = sum(1 for word in suspicious if word in text_lower)
    if count >= 2:
        return True, "Multiple suspicious keywords detected"
    
    return False, ""


def sanitize_text(text: str, max_length: int = 5000, field_name: str = "input") -> str:
    """
    ENHANCED: Sanitize text with prompt injection detection.
    """
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a non-empty string")
    
    # Length check
    if len(text) > max_length:
        raise HTTPException(
            status_code=400, 
            detail=f"{field_name} too long: {len(text)} chars (max: {max_length})"
        )
    
    # SECURITY: Prompt injection detection
    is_injection, reason = detect_prompt_injection(text)
    if is_injection:
        print(f"🚨 SECURITY ALERT: {reason} in {field_name}")
        raise HTTPException(
            status_code=400,
            detail=f"🚫 Invalid input detected: {reason}. Please provide only product information without instructions or commands."
        )
    
    # Remove control characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t\r')
    
    return text.strip()


def validate_image(file: UploadFile) -> bytes:
    """
    Validate uploaded image for security.
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    
    image_bytes = file.file.read()
    
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {len(image_bytes)} bytes (max: {MAX_SIZE})"
        )
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        if width * height > 25_000_000:
            raise HTTPException(
                status_code=400,
                detail=f"Image resolution too high: {width}x{height} (max: 25MP)"
            )
        
        allowed_formats = ['JPEG', 'PNG', 'WEBP', 'JPG']
        if img.format not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {img.format}. Allowed: {allowed_formats}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    return image_bytes


def validate_price(price: float) -> float:
    """Validate price is reasonable."""
    if not isinstance(price, (int, float)):
        raise HTTPException(status_code=400, detail="Price must be a number")
    
    if price < 0:
        raise HTTPException(status_code=400, detail="Price cannot be negative")
    
    if price > 1_000_000:
        raise HTTPException(status_code=400, detail="Price unreasonably high")
    
    return float(price)


def validate_category(category: str) -> str:
    """Validate category is from allowed list."""
    allowed_categories = [
        "Phone Case", "Furniture", "Clothing", "Electronics", 
        "Home Decor", "Toys", "Sports", "Other"
    ]
    
    category = sanitize_text(category, max_length=100, field_name="category")
    
    if category not in allowed_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed: {', '.join(allowed_categories)}"
        )
    
    return category


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Smart E-Commerce Assistant backend is running with security protection."}


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    SECURED: Analyze product with input validation and injection detection.
    """
    try:
        # SECURITY: Validate all inputs
        image_bytes = validate_image(file)
        clean_description = sanitize_text(description, max_length=2000, field_name="description")
        validated_price = validate_price(price)
        validated_category = validate_category(category)
        
        # Run analysis
        analysis = full_analysis(
            image_bytes=image_bytes,
            description=clean_description,
            price=validated_price,
            category=validated_category,
            generate_captions=True,
        )
        
        return analysis
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/improve_description")
async def improve_description_endpoint(
    description: str = Form(...),
    image_analysis: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    SECURED: Generate improved description with validation.
    """
    try:
        # SECURITY: Validate inputs
        clean_description = sanitize_text(description, max_length=2000, field_name="description")
        validated_price = validate_price(price)
        validated_category = validate_category(category)
        
        try:
            image_analysis_dict = json.loads(image_analysis)
            if not isinstance(image_analysis_dict, dict):
                raise ValueError("image_analysis must be a JSON object")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid image_analysis JSON")
        
        improved = generate_improved_description(
            image_analysis=image_analysis_dict,
            original_description=clean_description,
            price=validated_price,
            category=validated_category,
        )
        
        return {
            "original": clean_description,
            "improved": improved,
        }
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description improvement failed: {str(e)}")


@app.post("/generate_caption_from_suggestions")
async def generate_caption_from_suggestions_endpoint(
    original_caption: str = Form(...),
    suggestions: str = Form(...),
    image_analysis: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    SECURED: Generate improved caption with validation.
    """
    try:
        # SECURITY: Validate inputs
        clean_caption = sanitize_text(original_caption, max_length=500, field_name="caption")
        validated_price = validate_price(price)
        validated_category = validate_category(category)
        
        try:
            suggestions_list = json.loads(suggestions)
            image_analysis_dict = json.loads(image_analysis)
            
            if not isinstance(suggestions_list, list):
                raise ValueError("suggestions must be a JSON array")
            if not isinstance(image_analysis_dict, dict):
                raise ValueError("image_analysis must be a JSON object")
                
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        improved_caption = generate_caption_from_suggestions(
            original_caption=clean_caption,
            suggestions=suggestions_list,
            image_analysis=image_analysis_dict,
            price=validated_price,
            category=validated_category,
        )
        
        return {
            "original": clean_caption,
            "improved": improved_caption,
        }
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


@app.post("/reviews")
async def reviews_endpoint(
    caption: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    SECURED: Generate reviews with input validation.
    """
    try:
        # SECURITY: Validate inputs
        clean_caption = sanitize_text(caption, max_length=500, field_name="caption")
        clean_description = sanitize_text(description, max_length=2000, field_name="description")
        validated_price = validate_price(price)
        validated_category = validate_category(category)
        
        reviews = generate_reviews_for_product(
            caption=clean_caption,
            description=clean_description,
            price=validated_price,
            category=validated_category,
        )
        
        return {"reviews": reviews}
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review generation failed: {str(e)}")


@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    SECURED: Generate report with validation.
    """
    try:
        # SECURITY: Validate inputs
        image_bytes = validate_image(file)
        clean_description = sanitize_text(description, max_length=2000, field_name="description")
        validated_price = validate_price(price)
        validated_category = validate_category(category)
        
        analysis = full_analysis(
            image_bytes=image_bytes,
            description=clean_description,
            price=validated_price,
            category=validated_category,
            generate_captions=True,
        )
        
        image_analysis = analysis["image_analysis"]
        description_analysis = analysis["description_analysis"]
        comparison = analysis["comparison"]
        captions = analysis.get("captions", {})
        suggestions = analysis.get("suggestions", [])

        standard_caption = captions.get("standard", comparison.get("img_caption", ""))
        reviews = generate_reviews_for_product(
            caption=standard_caption,
            description=clean_description,
            price=validated_price,
            category=validated_category,
        )

        report_text = build_report_text(
            image_analysis=image_analysis,
            description=clean_description,
            description_analysis=description_analysis,
            comparison=comparison,
            reviews=reviews,
            price=validated_price,
            category=validated_category,
            captions=captions,
            suggestions=suggestions,
        )

        report_html = build_report_html(
            image_analysis=image_analysis,
            description=clean_description,
            description_analysis=description_analysis,
            comparison=comparison,
            reviews=reviews,
            price=validated_price,
            category=validated_category,
            captions=captions,
            suggestions=suggestions,
        )

        return {
            "report": report_text,
            "report_html": report_html,
            "analysis": analysis,
            "reviews": reviews,
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.post("/report_pdf")
async def report_pdf_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
):
    """
    SECURED: Generate PDF report with validation.
    """
    try:
        # SECURITY: Validate inputs
        image_bytes = validate_image(file)
        clean_description = sanitize_text(description, max_length=2000, field_name="description")
        validated_price = validate_price(price)
        validated_category = validate_category(category)

        analysis = full_analysis(
            image_bytes=image_bytes,
            description=clean_description,
            price=validated_price,
            category=validated_category,
            generate_captions=True,
        )
        
        image_analysis = analysis["image_analysis"]
        description_analysis = analysis["description_analysis"]
        comparison = analysis["comparison"]
        captions = analysis.get("captions", {})
        suggestions = analysis.get("suggestions", [])

        standard_caption = captions.get("standard", comparison.get("img_caption", ""))
        reviews = generate_reviews_for_product(
            caption=standard_caption,
            description=clean_description,
            price=validated_price,
            category=validated_category,
        )

        report_text = build_report_text(
            image_analysis=image_analysis,
            description=clean_description,
            description_analysis=description_analysis,
            comparison=comparison,
            reviews=reviews,
            price=validated_price,
            category=validated_category,
            captions=captions,
            suggestions=suggestions,
        )

        pdf_bytes = build_report_pdf(report_text=report_text, image_bytes=image_bytes)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="product_report.pdf"'},
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")