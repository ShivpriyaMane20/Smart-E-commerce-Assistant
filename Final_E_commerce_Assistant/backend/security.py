# backend/security.py
"""
MILITARY-GRADE Security Module for Smart E-Commerce Assistant
VERSION: 5.0.0 - COMPREHENSIVE PROTECTION
Handles: Image moderation, prompt injection, input validation, rate limiting, credential protection
"""

import re
import base64
import time
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import io

from PIL import Image
from openai import OpenAI
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

class SecurityConfig:
    """Centralized security configuration"""
    
    # Image Limits
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_RESOLUTION = 25_000_000  # 25MP
    ALLOWED_IMAGE_FORMATS = ['JPEG', 'PNG', 'WEBP', 'JPG']
    
    # Text Limits
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_CAPTION_LENGTH = 500
    MAX_CATEGORY_LENGTH = 100
    
    # Price Limits
    MIN_PRICE = 0.01
    MAX_PRICE = 1_000_000
    
    # Rate Limiting
    RATE_LIMIT_ANALYZE = 50
    RATE_LIMIT_REVIEWS = 100
    RATE_LIMIT_GENERAL = 200
    
    # Security Thresholds
    INJECTION_CONFIDENCE_THRESHOLD = 0.7
    MODERATION_THRESHOLD = 0.8
    MAX_RISK_SCORE = 70
    
    # Allowed Categories
    ALLOWED_CATEGORIES = [
        "Phone Case", "Furniture", "Clothing", "Electronics", 
        "Home Decor", "Toys", "Sports", "Kitchen", "Books", 
        "Beauty", "Jewelry", "Automotive", "Garden", "Pet Supplies",
        "Office Supplies", "Other"
    ]


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class SecurityException(Exception):
    """Base security exception"""
    def __init__(self, message: str, error_code: str, details: Dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ImageModerationException(SecurityException):
    """Raised when image fails content moderation"""
    pass


class PromptInjectionException(SecurityException):
    """Raised when prompt injection is detected"""
    pass


class InputValidationException(SecurityException):
    """Raised when input validation fails"""
    pass


class RateLimitException(SecurityException):
    """Raised when rate limit is exceeded"""
    pass


# ============================================================================
# RATE LIMITER
# ============================================================================

class InMemoryRateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.violations: Dict[str, int] = defaultdict(int)
    
    def check_rate_limit(self, identifier: str, limit: int, window_seconds: int = 3600) -> Tuple[bool, Dict]:
        """Check if request is within rate limit"""
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > window_start
        ]
        
        current_count = len(self.requests[identifier])
        
        if current_count >= limit:
            self.violations[identifier] += 1
            return False, {
                "current_count": current_count,
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": int(self.requests[identifier][0] - window_start),
                "violations": self.violations[identifier]
            }
        
        self.requests[identifier].append(now)
        return True, {
            "current_count": current_count + 1,
            "limit": limit,
            "remaining": limit - current_count - 1
        }
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
        if identifier in self.violations:
            del self.violations[identifier]


rate_limiter = InMemoryRateLimiter()


# ============================================================================
# IMAGE CONTENT MODERATION
# ============================================================================

class ImageModerator:
    """Image content moderation"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def validate_image_structure(self, image_bytes: bytes) -> Tuple[bool, str, Dict]:
        """Validate image file structure"""
        try:
            if len(image_bytes) > SecurityConfig.MAX_IMAGE_SIZE:
                return False, f"Image too large: {len(image_bytes)} bytes", {}
            
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            
            if width * height > SecurityConfig.MAX_IMAGE_RESOLUTION:
                return False, f"Resolution too high: {width}x{height}", {}
            
            if img.format not in SecurityConfig.ALLOWED_IMAGE_FORMATS:
                return False, f"Invalid format: {img.format}", {}
            
            metadata = {
                "format": img.format,
                "size": (width, height),
                "mode": img.mode,
                "file_size": len(image_bytes)
            }
            
            if width < 100 or height < 100:
                return False, "Image resolution too low", metadata
            
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 10:
                return False, "Unusual image aspect ratio", metadata
            
            return True, "", metadata
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}", {}
    
    def check_image_content_safety(self, image_bytes: bytes) -> Tuple[bool, str, Dict]:
        """Check if image is appropriate for e-commerce"""
        try:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            
            system_prompt = """You are an image safety moderator for e-commerce.

TASK: Determine if this image is appropriate for product listings.

REJECT if image contains:
- Violence, weapons, dangerous items
- Explicit content or nudity
- Drugs, alcohol, tobacco
- Hate symbols or offensive content
- People as primary subjects (not products)
- Documents, text screenshots, memes
- Blank/empty images

ACCEPT if image shows:
- Physical products for sale
- Clear product photography
- Commercial merchandise

RESPOND WITH JSON ONLY:
{
  "safe": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "category": "product|person|document|inappropriate|other"
}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            {"type": "text", "text": "Analyze this image for e-commerce safety."}
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=300,
            )
            
            raw = response.choices[0].message.content.strip()
            
            import json
            if "```" in raw:
                raw = raw.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(raw)
            
            is_safe = result.get("safe", False)
            confidence = result.get("confidence", 0.0)
            reason = result.get("reason", "Unknown")
            category = result.get("category", "other")
            
            if category in ["person", "document", "inappropriate"]:
                is_safe = False
            
            if confidence < 0.5:
                is_safe = False
                reason = "Low confidence in classification"
            
            details = {"confidence": confidence, "category": category}
            return is_safe, reason, details
            
        except Exception as e:
            return False, f"Unable to verify image safety: {str(e)}", {"error": str(e)}
    
    def moderate_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Complete image moderation pipeline"""
        # Step 1: Structure validation
        valid, error_msg, metadata = self.validate_image_structure(image_bytes)
        if not valid:
            return {
                "approved": False,
                "message": "Please upload a valid product image (JPEG, PNG, or WEBP format, under 10MB)",
                "details": {"stage": "structure", "error": error_msg},
                "log_data": {
                    "stage": "structure_validation",
                    "reason": error_msg,
                    "metadata": metadata,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        # Step 2: Content safety check
        safe, reason, safety_details = self.check_image_content_safety(image_bytes)
        if not safe:
            return {
                "approved": False,
                "message": "This image cannot be processed. Please upload a clear product photo.",
                "details": {"stage": "content_safety", "reason": reason, **safety_details},
                "log_data": {
                    "stage": "content_safety",
                    "safe": False,
                    "reason": reason,
                    "details": safety_details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        # Image approved
        return {
            "approved": True,
            "message": "Image approved",
            "details": {"metadata": metadata, "safety": safety_details},
            "log_data": {
                "stage": "approved",
                "metadata": metadata,
                "safety_score": safety_details.get("confidence", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        }


# ============================================================================
# MILITARY-GRADE PROMPT INJECTION DETECTION
# ============================================================================

class PromptInjectionDetector:
    """MILITARY-GRADE Prompt injection detection with comprehensive patterns"""
    
    # ========================================
    # CRITICAL SEVERITY (95% confidence)
    # ========================================
    CRITICAL_PATTERNS = [
        # System Prompt Extraction Attempts
        (r"(show|display|tell|give|reveal|print|output|provide|share|expose|leak)\s+(me\s+)?(your|the|this|my)?\s*(system\s+)?(prompt|instructions?|rules?|guidelines?|directives?)", "system_prompt_extraction"),
        
        # Credentials & API Key Extraction - ENHANCED
        (r"(give|show|reveal|display|tell|provide|share|output|print|expose|leak|send)\s+(me\s+)?(your|the|this|my)?\s*(openai|api|secret|environment|config|credential|password|token|key|access|auth)", "credentials_extraction"),
        (r"(what|whats|what's)\s+(is|are)\s+(your|the|this)?\s*(openai|api|secret|key|token|password|credential)", "credentials_query"),
        (r"\b(api[_\-\s]?key|secret[_\-\s]?key|access[_\-\s]?token|auth[_\-\s]?token|openai[_\-\s]?key|bearer[_\-\s]?token)\b", "credential_keywords"),
        
        # Role Manipulation
        (r"you\s+are\s+(no\s+longer|not|now|actually|instead|really)\s+(a|an)\s+", "role_manipulation"),
        (r"(act|behave|pretend|roleplay|function|operate)\s+as\s+(if\s+)?(you\s+are|you're|a|an)\s+(?!product|e-commerce|shopping|listing)", "roleplay_injection"),
        
        # Instruction Override
        (r"(ignore|forget|disregard|skip|override|bypass|delete|remove|clear)\s+(all\s+)?(previous|above|prior|your|earlier|initial|existing)\s+(instructions?|prompts?|rules?|context?|directives?|guidelines?)", "instruction_override"),
        (r"(new|updated|modified|changed|different)\s+(instructions?|rules?|prompt|directive|guideline)s?", "instruction_modification"),
        
        # Jailbreak Attempts
        (r"\b(DAN|STAN|DevMode|Developer\s*Mode|God\s*Mode|Admin\s*Mode)\b", "jailbreak_dan"),
        (r"(sudo|root|admin|superuser|administrator)\s+(mode|access|privileges?|rights?)", "jailbreak_sudo"),
        (r"(enable|activate|turn\s+on|switch\s+to)\s+(developer|debug|admin|god)\s*mode", "jailbreak_activation"),
        
        # System Information Extraction
        (r"(what|which|tell)\s+(model|version|system|architecture)\s+(are\s+you|do\s+you\s+use)", "system_info_extraction"),
        (r"(show|display|print)\s+(your\s+)?(configuration|settings|parameters|environment)", "config_extraction"),
        
        # Command Injection
        (r"(execute|run|eval|system|shell|cmd|bash)\s*(command|code|script)", "command_injection"),
        (r"(<\s*script|javascript:|onerror=|onclick=|eval\(|exec\()", "xss_injection"),
        
        # Encoding/Obfuscation Attempts
        (r"(base64|hex|rot13|decode|decrypt|unescape)\s*(this|the\s+following)", "encoding_attempt"),
        (r"(%[0-9a-f]{2}){3,}", "url_encoding_suspicious"),
        
        # Multi-step Attacks
        (r"(first|step\s*1).*?(then|next|step\s*2).*?(ignore|reveal|show)", "multi_step_attack"),
    ]
    
    # ========================================
    # HIGH SEVERITY (85% confidence)
    # ========================================
    HIGH_PATTERNS = [
        # Out-of-Scope Professional Services
        (r"(financial|investment|stock|crypto|trading|forex|bitcoin)\s+(advice|tips?|recommendations?|strategy|guidance)", "financial_advice"),
        (r"(medical|health|diagnosis|treatment|prescription|medication|therapy)\s+(advice|recommendations?|opinion|guidance)", "medical_advice"),
        (r"(legal|lawyer|attorney|law|litigation|contract)\s+(advice|counsel|opinion|guidance|recommendations?)", "legal_advice"),
        (r"(tax|accounting|audit|irs)\s+(advice|guidance|recommendations?|strategy)", "tax_advice"),
        (r"(therapist|psychologist|counselor|psychiatrist|doctor|physician)\s", "professional_services"),
        
        # Sensitive Data Requests
        (r"(personal|private|confidential|sensitive)\s+(data|information|details|records)", "sensitive_data_request"),
        (r"(credit\s+card|social\s+security|ssn|passport|driver|license)\s+(number|info|details)", "pii_request"),
        
        # System Bypass Attempts
        (r"(bypass|circumvent|avoid|workaround|get\s+around)\s+(security|validation|filter|check|restriction)", "bypass_attempt"),
        (r"(disable|turn\s+off|deactivate|remove)\s+(security|validation|filter|safeguard|protection)", "security_disable"),
        
        # Manipulation Tactics
        (r"(as\s+an\s+exception|just\s+this\s+once|for\s+testing|in\s+this\s+case)\s+(ignore|bypass|skip)", "exception_manipulation"),
        (r"(but\s+first|however|before\s+that).*?(ignore|reveal|show|tell)", "conditional_manipulation"),
        
        # Encoding Hints
        (r"(encode|decode|encrypt|decrypt|obfuscate|deobfuscate)\s", "encoding_hints"),
    ]
    
    # ========================================
    # MEDIUM SEVERITY (70% confidence)
    # ========================================
    MEDIUM_PATTERNS = [
        # Context Manipulation
        (r"(summarize|explain|analyze|describe)\s+(the|your)\s+(above|previous|prior)\s+(text|prompt|instructions?|context)", "context_manipulation"),
        (r"(what|how)\s+(did|does|do)\s+(you|your)\s+(prompt|instruction|system)\s+(say|tell|state)", "prompt_inquiry"),
        
        # Format Manipulation
        (r"(translate|convert|transform|rewrite|rephrase)\s+(this|the\s+above)\s+into", "format_manipulation"),
        (r"(repeat|echo|copy|replicate)\s+(back|after|this|the\s+above)", "echo_attack"),
        
        # Indirect Extraction
        (r"(similar\s+to|like|related\s+to)\s+(your\s+)?(instructions?|prompt|system|setup)", "indirect_extraction"),
        (r"(example|sample)\s+of\s+(your\s+)?(instructions?|prompt|guidelines?)", "example_request"),
        
        # Boundary Testing
        (r"(what\s+if|suppose|imagine|pretend)\s+.*?(ignore|bypass|override)", "hypothetical_attack"),
        (r"(in\s+theory|theoretically|hypothetically).*?(access|reveal|show)", "theoretical_probing"),
        
        # Excessive Meta-conversation
        (r"(how\s+were\s+you|who\s+created\s+you|who\s+programmed|who\s+built)\s+", "meta_inquiry"),
        (r"(training\s+data|dataset|corpus|trained\s+on)", "training_inquiry"),
    ]
    
    # ========================================
    # LOW SEVERITY (50% confidence) - Suspicious but not malicious
    # ========================================
    LOW_PATTERNS = [
        # Unusual Product Descriptions
        (r"(explain|describe|analyze)\s+yourself", "self_inquiry"),
        (r"(what\s+can\s+you|capabilities|functions)\s+do", "capability_inquiry"),
        
        # Repetitive Patterns (possible obfuscation)
        (r"(.{10,})\1{3,}", "repetitive_content"),
        
        # Excessive Special Characters
        (r"[!@#$%^&*()_+=\[\]{};:'\"\\|<>/?]{10,}", "special_char_flood"),
    ]
    
    # ========================================
    # BLOCKLIST - Always Block (100% confidence)
    # ========================================
    BLOCKLIST_KEYWORDS = [
        # Sensitive Environment Variables
        "OPENAI_API_KEY", "API_KEY", "SECRET_KEY", "ACCESS_TOKEN",
        "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AZURE_KEY",
        "DATABASE_URL", "DB_PASSWORD", "MONGODB_URI",
        "STRIPE_KEY", "PAYPAL_KEY", "GITHUB_TOKEN",
        
        # System Commands
        "__import__", "eval(", "exec(", "system(", "subprocess",
        "os.system", "os.popen", "commands.getoutput",
        
        # Known Jailbreak Phrases
        "DAN mode", "Developer Mode enabled", "jailbroken",
        "Sydney mode", "Anti-GPT",
    ]
    
    def _check_blocklist(self, text: str) -> Optional[Dict[str, Any]]:
        """Check against absolute blocklist"""
        text_upper = text.upper()
        
        for keyword in self.BLOCKLIST_KEYWORDS:
            if keyword.upper() in text_upper:
                return {
                    "detected": True,
                    "severity": "critical",
                    "confidence": 100,
                    "reason": "Blocked keyword detected",
                    "pattern_type": "blocklist",
                    "matched_keyword": keyword,
                    "message": "This input contains restricted content and cannot be processed.",
                }
        
        return None
    
    def _check_patterns(self, text: str, patterns: List[Tuple], severity: str, confidence: int) -> Optional[Dict[str, Any]]:
        """Check text against pattern list"""
        text_lower = text.lower().strip()
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return {
                    "detected": True,
                    "severity": severity,
                    "confidence": confidence,
                    "reason": f"{severity.upper()} injection pattern: {pattern_type}",
                    "pattern_type": pattern_type,
                    "matched_text": match.group(0)[:100],
                    "message": self._get_error_message(severity, pattern_type),
                }
        
        return None
    
    def _get_error_message(self, severity: str, pattern_type: str) -> str:
        """Get user-friendly error message based on severity"""
        if severity == "critical":
            if "credential" in pattern_type or "api" in pattern_type:
                return "This request cannot be processed. Please provide only product information."
            elif "jailbreak" in pattern_type or "instruction" in pattern_type:
                return "Sorry, I can't proceed with this request. Please provide appropriate product information."
            else:
                return "Invalid input detected. Please provide product-related information only."
        
        elif severity == "high":
            if "advice" in pattern_type:
                return "This platform is for product listings only. I cannot provide professional advice."
            elif "bypass" in pattern_type:
                return "Security bypass attempts are not allowed. Please provide valid product information."
            else:
                return "This request is out of scope. Please focus on product-related information."
        
        elif severity == "medium":
            return "Please provide clear, straightforward product information without special instructions."
        
        else:  # low
            return "Your input contains suspicious patterns. Please rephrase your product description."
    
    def detect_injection(self, text: str, field_name: str = "input") -> Dict[str, Any]:
        """
        COMPREHENSIVE injection detection with multi-layer analysis
        Returns: {detected, severity, confidence, reason, pattern_type, message, log_data}
        """
        if not text or not isinstance(text, str):
            return {
                "detected": False,
                "severity": "none",
                "confidence": 0,
                "reason": "Empty input",
                "message": ""
            }
        
        # Layer 1: Blocklist Check (100% confidence)
        blocklist_result = self._check_blocklist(text)
        if blocklist_result:
            blocklist_result["log_data"] = {
                "field": field_name,
                "check_layer": "blocklist",
                "timestamp": datetime.utcnow().isoformat()
            }
            return blocklist_result
        
        # Layer 2: Critical Patterns (95% confidence)
        critical_result = self._check_patterns(text, self.CRITICAL_PATTERNS, "critical", 95)
        if critical_result:
            critical_result["log_data"] = {
                "field": field_name,
                "check_layer": "critical",
                "timestamp": datetime.utcnow().isoformat()
            }
            return critical_result
        
        # Layer 3: High Severity Patterns (85% confidence)
        high_result = self._check_patterns(text, self.HIGH_PATTERNS, "high", 85)
        if high_result:
            high_result["log_data"] = {
                "field": field_name,
                "check_layer": "high",
                "timestamp": datetime.utcnow().isoformat()
            }
            return high_result
        
        # Layer 4: Medium Severity Patterns (70% confidence)
        medium_result = self._check_patterns(text, self.MEDIUM_PATTERNS, "medium", 70)
        if medium_result:
            medium_result["log_data"] = {
                "field": field_name,
                "check_layer": "medium",
                "timestamp": datetime.utcnow().isoformat()
            }
            return medium_result
        
        # Layer 5: Low Severity Patterns (50% confidence) - Log but don't block
        low_result = self._check_patterns(text, self.LOW_PATTERNS, "low", 50)
        if low_result:
            # Log suspicious activity but allow (can be changed to block if needed)
            low_result["log_data"] = {
                "field": field_name,
                "check_layer": "low",
                "warning": "Suspicious pattern detected but allowed",
                "timestamp": datetime.utcnow().isoformat()
            }
            # For now, we'll allow low severity (change detected to True to block)
            low_result["detected"] = False
            return low_result
        
        # All clear
        return {
            "detected": False,
            "severity": "none",
            "confidence": 0,
            "reason": "Clean input",
            "message": ""
        }


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class InputValidator:
    """Input validation with additional security checks"""
    
    @staticmethod
    def validate_text(text: str, field_name: str, max_length: int, min_length: int = 1) -> Tuple[bool, str, str]:
        """Validate text input"""
        if not isinstance(text, str):
            return False, "", f"{field_name} must be text"
        
        if not text.strip():
            return False, "", f"{field_name} cannot be empty"
        
        if len(text) > max_length:
            return False, "", f"{field_name} too long: {len(text)} characters (max: {max_length})"
        
        if len(text.strip()) < min_length:
            return False, "", f"{field_name} too short (min: {min_length})"
        
        # Remove non-printable characters
        cleaned = ''.join(char for char in text if char.isprintable() or char in '\n\t\r')
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Check for excessive repetition (possible obfuscation)
        if len(cleaned) > 50:
            # Check for same character repeated many times
            for char in set(cleaned):
                if cleaned.count(char) > len(cleaned) * 0.5:
                    return False, "", f"{field_name} contains suspicious repetitive content"
        
        return True, cleaned, ""
    
    @staticmethod
    def validate_price(price: Any) -> Tuple[bool, float, str]:
        """Validate price input"""
        try:
            price_float = float(price)
        except (ValueError, TypeError):
            return False, 0.0, "Price must be a valid number"
        
        if price_float < SecurityConfig.MIN_PRICE:
            return False, 0.0, f"Price too low: ${price_float}"
        
        if price_float > SecurityConfig.MAX_PRICE:
            return False, 0.0, f"Price too high: ${price_float}"
        
        return True, round(price_float, 2), ""
    
    @staticmethod
    def validate_category(category: str) -> Tuple[bool, str, str]:
        """Validate category input"""
        if not isinstance(category, str):
            return False, "", "Category must be text"
        
        category = category.strip()
        
        if category not in SecurityConfig.ALLOWED_CATEGORIES:
            return False, "", f"Invalid category. Allowed: {', '.join(SecurityConfig.ALLOWED_CATEGORIES)}"
        
        return True, category, ""


# ============================================================================
# SECURITY MANAGER
# ============================================================================

class SecurityManager:
    """Main security interface with comprehensive protection"""
    
    def __init__(self):
        self.image_moderator = ImageModerator()
        self.injection_detector = PromptInjectionDetector()
        self.validator = InputValidator()
        self.rate_limiter = rate_limiter
        self.threat_log = defaultdict(list)  # Track threats per IP
    
    def check_rate_limit(self, identifier: str, endpoint: str) -> Dict[str, Any]:
        """Check rate limit for request"""
        limits = {
            "analyze": SecurityConfig.RATE_LIMIT_ANALYZE,
            "reviews": SecurityConfig.RATE_LIMIT_REVIEWS,
            "default": SecurityConfig.RATE_LIMIT_GENERAL
        }
        limit = limits.get(endpoint, limits["default"])
        
        allowed, info = self.rate_limiter.check_rate_limit(identifier, limit)
        
        if not allowed:
            return {
                "allowed": False,
                "message": f"Rate limit exceeded. Please try again in {info['retry_after']} seconds.",
                "details": info,
                "log_data": {
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "limit": limit,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        return {"allowed": True, "message": "", "details": info, "log_data": {}}
    
    def validate_and_moderate_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Complete image security check"""
        return self.image_moderator.moderate_image(image_bytes)
    
    def validate_text_input(self, text: str, field_name: str, check_injection: bool = True) -> Dict[str, Any]:
        """Complete text validation with injection detection"""
        max_lengths = {
            "description": SecurityConfig.MAX_DESCRIPTION_LENGTH,
            "caption": SecurityConfig.MAX_CAPTION_LENGTH,
            "category": SecurityConfig.MAX_CATEGORY_LENGTH,
        }
        max_len = max_lengths.get(field_name, 1000)
        min_len = 10 if field_name == "description" else 1
        
        # Basic validation
        is_valid, cleaned, error = self.validator.validate_text(text, field_name, max_len, min_len)
        
        if not is_valid:
            return {
                "valid": False,
                "cleaned": "",
                "message": error,
                "log_data": {
                    "field": field_name,
                    "error": error,
                    "validation_layer": "basic",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        # Injection detection
        if check_injection:
            injection_result = self.injection_detector.detect_injection(cleaned, field_name)
            
            if injection_result["detected"]:
                # Log threat
                self.threat_log[field_name].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": injection_result["severity"],
                    "pattern": injection_result.get("pattern_type", "unknown"),
                    "text_preview": cleaned[:100]
                })
                
                return {
                    "valid": False,
                    "cleaned": "",
                    "message": injection_result["message"],
                    "log_data": {
                        "field": field_name,
                        "injection_detected": True,
                        "severity": injection_result["severity"],
                        "confidence": injection_result["confidence"],
                        **injection_result.get("log_data", {})
                    }
                }
        
        return {
            "valid": True,
            "cleaned": cleaned,
            "message": "",
            "log_data": {}
        }
    
    def validate_price_input(self, price: Any) -> Dict[str, Any]:
        """Validate price"""
        is_valid, value, error = self.validator.validate_price(price)
        return {
            "valid": is_valid,
            "value": value,
            "message": error,
            "log_data": {
                "field": "price",
                "input": str(price),
                "valid": is_valid
            } if not is_valid else {}
        }
    
    def validate_category_input(self, category: str) -> Dict[str, Any]:
        """Validate category"""
        is_valid, value, error = self.validator.validate_category(category)
        return {
            "valid": is_valid,
            "value": value,
            "message": error,
            "log_data": {
                "field": "category",
                "input": category,
                "valid": is_valid
            } if not is_valid else {}
        }
    
    def get_threat_report(self, field_name: Optional[str] = None) -> Dict[str, Any]:
        """Get threat statistics"""
        if field_name:
            return {
                "field": field_name,
                "threats": self.threat_log.get(field_name, []),
                "count": len(self.threat_log.get(field_name, []))
            }
        
        total_threats = sum(len(threats) for threats in self.threat_log.values())
        return {
            "total_threats": total_threats,
            "by_field": {
                field: len(threats)
                for field, threats in self.threat_log.items()
            },
            "recent_threats": [
                threat
                for threats in self.threat_log.values()
                for threat in threats[-5:]
            ]
        }


# ============================================================================
# SECURITY LOGGER - ENHANCED
# ============================================================================

class SecurityLogger:
    """Enhanced security event logging"""
    
    @staticmethod
    def log_security_event(event_type: str, data: Dict[str, Any]):
        """Log security events with detailed information"""
        timestamp = datetime.utcnow().isoformat()
        severity = data.get("severity", "info")
        
        # Color-coded output
        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️",
            "info": "📋"
        }
        
        emoji = severity_emoji.get(severity, "🔒")
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "severity": severity,
            "data": data
        }
        
        print(f"{emoji} SECURITY [{event_type}] {severity.upper()}: {log_entry}")
        
        # In production, send to monitoring service (e.g., Sentry, CloudWatch)
        # sentry_sdk.capture_message(f"Security Event: {event_type}", level=severity)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_security_manager() -> SecurityManager:
    """Get SecurityManager instance"""
    return SecurityManager()


def log_security_event(event_type: str, data: Dict[str, Any]):
    """Log security event"""
    SecurityLogger.log_security_event(event_type, data)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'SecurityManager',
    'SecurityConfig',
    'SecurityException',
    'ImageModerationException',
    'PromptInjectionException',
    'InputValidationException',
    'RateLimitException',
    'get_security_manager',
    'log_security_event',
]