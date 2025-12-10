# backend/api.py
# OPTIMIZED: Fast, secure, no prompts.py dependency

import io
from typing import Any, Dict
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .core import (
    full_analysis,
    generate_reviews_for_product,
    build_report_text,
    build_report_html,
    build_report_pdf,
    generate_improved_description,
    generate_caption_from_suggestions,
)

from .security import (
    get_security_manager,
    log_security_event,
    SecurityException,
    ImageModerationException,
    PromptInjectionException,
    InputValidationException,
    RateLimitException,
    SecurityConfig,
)


app = FastAPI(
    title="Smart E-Commerce Assistant API",
    description="AI-powered product listing analyzer with security",
    version="4.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security_manager = get_security_manager()


# ============================================================================
# UTILITIES
# ============================================================================

def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    return client_ip


def create_error_response(status_code: int, message: str, error_code: str = "ERROR") -> JSONResponse:
    """Create standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "code": error_code,
                "timestamp": json.loads(json.dumps({"t": None}, default=str))
            }
        }
    )


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(SecurityException)
async def security_exception_handler(request: Request, exc: SecurityException):
    """Handle security exceptions"""
    log_security_event(
        event_type=f"security_exception_{exc.error_code.lower()}",
        data={
            "path": str(request.url),
            "client": get_client_identifier(request),
            "error_code": exc.error_code,
            "message": exc.message
        }
    )
    
    status_codes = {
        "ImageModerationException": 400,
        "PromptInjectionException": 400,
        "InputValidationException": 400,
        "RateLimitException": 429,
    }
    status_code = status_codes.get(exc.__class__.__name__, 400)
    
    return create_error_response(status_code, exc.message, exc.error_code)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return create_error_response(exc.status_code, exc.detail, "HTTP_ERROR")


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    log_security_event("unhandled_exception", {
        "path": str(request.url),
        "client": get_client_identifier(request),
        "error": str(exc)
    })
    
    return create_error_response(
        500, 
        "An error occurred. Please try again later.", 
        "INTERNAL_ERROR"
    )


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_and_check_rate_limit(request: Request, endpoint: str):
    """Check rate limit"""
    client_id = get_client_identifier(request)
    rate_result = security_manager.check_rate_limit(client_id, endpoint)
    
    if not rate_result["allowed"]:
        log_security_event("rate_limit_exceeded", rate_result["log_data"])
        raise RateLimitException(
            message=rate_result["message"],
            error_code="RATE_LIMIT_EXCEEDED",
            details=rate_result["details"]
        )


def validate_image_security(file: UploadFile) -> bytes:
    """Validate image security"""
    try:
        image_bytes = file.file.read()
    except Exception as e:
        log_security_event("image_read_error", {"error": str(e)})
        raise InputValidationException(
            message="Unable to read image file. Please try again.",
            error_code="IMAGE_READ_ERROR"
        )
    
    result = security_manager.validate_and_moderate_image(image_bytes)
    
    if not result["approved"]:
        log_security_event("image_rejected", result["log_data"])
        raise ImageModerationException(
            message=result["message"],
            error_code="IMAGE_NOT_APPROVED",
            details=result["details"]
        )
    
    log_security_event("image_approved", {
        "filename": file.filename,
        "size": len(image_bytes)
    })
    
    return image_bytes


def validate_text_security(text: str, field_name: str) -> str:
    """Validate text security"""
    result = security_manager.validate_text_input(text, field_name, check_injection=True)
    
    if not result["valid"]:
        log_security_event("input_rejected", result["log_data"])
        raise InputValidationException(
            message=result["message"],
            error_code="INPUT_VALIDATION_FAILED",
            details={"field": field_name}
        )
    
    return result["cleaned"]


def validate_price_security(price: float) -> float:
    """Validate price"""
    result = security_manager.validate_price_input(price)
    
    if not result["valid"]:
        log_security_event("price_validation_failed", result["log_data"])
        raise InputValidationException(
            message=result["message"],
            error_code="INVALID_PRICE"
        )
    
    return result["value"]


def validate_category_security(category: str) -> str:
    """Validate category"""
    result = security_manager.validate_category_input(category)
    
    if not result["valid"]:
        log_security_event("category_validation_failed", result["log_data"])
        raise InputValidationException(
            message=result["message"],
            error_code="INVALID_CATEGORY"
        )
    
    return result["value"]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root() -> Dict[str, str]:
    """Health check"""
    return {
        "message": "Smart E-Commerce Assistant API",
        "version": "4.1.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Detailed health check"""
    return {
        "status": "healthy",
        "security_features": ["image_moderation", "injection_detection", "rate_limiting"]
    }


@app.get("/categories")
async def get_categories() -> Dict[str, Any]:
    """Get available categories"""
    return {
        "categories": SecurityConfig.ALLOWED_CATEGORIES,
        "total": len(SecurityConfig.ALLOWED_CATEGORIES)
    }


@app.post("/analyze")
async def analyze_endpoint(
    request: Request,
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    Analyze product with security checks
    """
    try:
        # Security checks
        validate_and_check_rate_limit(request, "analyze")
        image_bytes = validate_image_security(file)
        clean_description = validate_text_security(description, "description")
        validated_price = validate_price_security(price)
        validated_category = validate_category_security(category)
        
        # Analysis
        analysis = full_analysis(
            image_bytes=image_bytes,
            description=clean_description,
            price=validated_price,
            category=validated_category,
            generate_captions=True,
        )
        
        log_security_event("analysis_success", {
            "category": validated_category,
            "price": validated_price,
            "risk_score": analysis.get("comparison", {}).get("risk_score", 0)
        })
        
        return analysis
        
    except SecurityException:
        raise
    except Exception as e:
        log_security_event("analysis_error", {"error": str(e)})
        raise HTTPException(500, "Unable to complete analysis. Please try again.")


@app.post("/improve_description")
async def improve_description_endpoint(
    request: Request,
    description: str = Form(...),
    image_analysis: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """Generate improved description"""
    try:
        validate_and_check_rate_limit(request, "default")
        
        clean_description = validate_text_security(description, "description")
        validated_price = validate_price_security(price)
        validated_category = validate_category_security(category)
        
        try:
            image_analysis_dict = json.loads(image_analysis)
            if not isinstance(image_analysis_dict, dict):
                raise ValueError("Invalid image_analysis")
        except (json.JSONDecodeError, ValueError):
            raise InputValidationException("Invalid image analysis data.", "INVALID_JSON")
        
        improved = generate_improved_description(
            image_analysis=image_analysis_dict,
            original_description=clean_description,
            price=validated_price,
            category=validated_category,
        )
        
        return {
            "original": clean_description,
            "improved": improved,
            "category": validated_category
        }
        
    except SecurityException:
        raise
    except Exception as e:
        log_security_event("description_improvement_error", {"error": str(e)})
        raise HTTPException(500, "Unable to generate improved description.")


@app.post("/generate_caption_from_suggestions")
async def generate_caption_from_suggestions_endpoint(
    request: Request,
    original_caption: str = Form(...),
    suggestions: str = Form(...),
    image_analysis: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """Generate improved caption"""
    try:
        validate_and_check_rate_limit(request, "default")
        
        clean_caption = validate_text_security(original_caption, "caption")
        validated_price = validate_price_security(price)
        validated_category = validate_category_security(category)
        
        try:
            suggestions_list = json.loads(suggestions)
            image_analysis_dict = json.loads(image_analysis)
            
            if not isinstance(suggestions_list, list):
                raise ValueError("suggestions must be array")
            if not isinstance(image_analysis_dict, dict):
                raise ValueError("image_analysis must be object")
                
        except (json.JSONDecodeError, ValueError):
            raise InputValidationException("Invalid JSON data.", "INVALID_JSON")
        
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
            "category": validated_category
        }
        
    except SecurityException:
        raise
    except Exception as e:
        log_security_event("caption_generation_error", {"error": str(e)})
        raise HTTPException(500, "Unable to generate improved caption.")


@app.post("/reviews")
async def reviews_endpoint(
    request: Request,
    caption: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """Generate customer reviews"""
    try:
        validate_and_check_rate_limit(request, "reviews")
        
        clean_caption = validate_text_security(caption, "caption")
        clean_description = validate_text_security(description, "description")
        validated_price = validate_price_security(price)
        validated_category = validate_category_security(category)
        
        reviews = generate_reviews_for_product(
            caption=clean_caption,
            description=clean_description,
            price=validated_price,
            category=validated_category,
        )
        
        log_security_event("reviews_generated", {
            "category": validated_category,
            "review_count": len(reviews)
        })
        
        return {"reviews": reviews, "category": validated_category}
        
    except SecurityException:
        raise
    except Exception as e:
        log_security_event("review_generation_error", {"error": str(e)})
        raise HTTPException(500, "Unable to generate reviews.")


@app.post("/report")
async def report_endpoint(
    request: Request,
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """Generate comprehensive report"""
    try:
        validate_and_check_rate_limit(request, "analyze")
        
        image_bytes = validate_image_security(file)
        clean_description = validate_text_security(description, "description")
        validated_price = validate_price_security(price)
        validated_category = validate_category_security(category)
        
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

        log_security_event("report_generated", {
            "category": validated_category,
            "risk_score": comparison.get("risk_score", 0)
        })

        return {
            "report": report_text,
            "report_html": report_html,
            "analysis": analysis,
            "reviews": reviews,
        }

    except SecurityException:
        raise
    except Exception as e:
        log_security_event("report_generation_error", {"error": str(e)})
        raise HTTPException(500, "Unable to generate report.")


@app.post("/report_pdf")
async def report_pdf_endpoint(
    request: Request,
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
):
    """Generate PDF report"""
    try:
        validate_and_check_rate_limit(request, "analyze")
        
        image_bytes = validate_image_security(file)
        clean_description = validate_text_security(description, "description")
        validated_price = validate_price_security(price)
        validated_category = validate_category_security(category)

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

        log_security_event("pdf_generated", {
            "category": validated_category,
            "size": len(pdf_bytes)
        })

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="product_report_{validated_category}.pdf"'},
        )

    except SecurityException:
        raise
    except Exception as e:
        log_security_event("pdf_generation_error", {"error": str(e)})
        raise HTTPException(500, "Unable to generate PDF.")


@app.get("/security/status")
async def security_status() -> Dict[str, Any]:
    """Get security system status"""
    return {
        "status": "active",
        "features": {
            "image_moderation": "enabled",
            "prompt_injection_detection": "enabled",
            "input_validation": "enabled",
            "rate_limiting": "enabled"
        },
        "version": "4.1.0"
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Smart E-Commerce Assistant API v4.1.0")
    print("🛡️  Security: ENABLED")
    print("⚡ Optimized for speed")
    
    log_security_event("api_startup", {"version": "4.1.0"})


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down API")
    log_security_event("api_shutdown", {})