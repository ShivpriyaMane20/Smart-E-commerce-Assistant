# streamlit_app.py - Merged Smart E-Commerce Assistant
# All-in-one deployment for Streamlit Cloud

import streamlit as st
import io
import os
import json
import math
import base64
from typing import Any, Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get OpenAI API key from Streamlit secrets or environment
def get_openai_client() -> OpenAI:
    """Get configured OpenAI client with API key."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        # Fallback to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Add it to Streamlit secrets or .env file")
        st.stop()
    
    return OpenAI(api_key=api_key)

VISION_MODEL = "gpt-4o-mini"
TEXT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ============================================================================
# CORE AI FUNCTIONS (from core.py)
# ============================================================================

def sanitize_text_input(text: str, max_length: int = 5000) -> str:
    """Sanitize user input to prevent prompt injection attacks."""
    if not text or not isinstance(text, str):
        return ""
    
    text = text[:max_length]
    
    injection_patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard above",
        "forget everything",
        "system:",
    ]
    
    text_lower = text.lower()
    for pattern in injection_patterns:
        if pattern in text_lower:
            text = text.replace(pattern, "[FILTERED]")
    
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    return text.strip()


def analyze_image(image_bytes: bytes) -> Dict[str, Any]:
    """Analyze product image using GPT-4 Vision."""
    client = get_openai_client()
    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = """You are an expert e-commerce product image analyzer.

Analyze the product image and return structured data in JSON format.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "caption": "brief factual product description (10-15 words)",
  "color": "primary color name or null",
  "material": "material type or null",
  "product_type": "product category or null",
  "style": "design style or null",
  "visible_features": ["list of clearly visible features"]
}"""

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        }
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )

        raw = resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        
        required_keys = ["caption", "color", "material", "product_type", "style", "visible_features"]
        for key in required_keys:
            if key not in parsed:
                parsed[key] = None if key != "visible_features" else []
        
        return parsed
        
    except Exception as e:
        return {
            "caption": "Unable to analyze image",
            "color": None,
            "material": None,
            "product_type": None,
            "style": None,
            "visible_features": [],
            "error": str(e)
        }


def generate_multiple_captions(image_bytes: bytes) -> Dict[str, Any]:
    """Generate multiple caption styles for different marketing purposes."""
    client = get_openai_client()
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    
    system_prompt = """You are an expert e-commerce copywriter.

Generate 3 caption variations optimized for different purposes.

Return ONLY valid JSON:
{
  "captions": {
    "standard": "Professional factual caption (10-15 words)",
    "enhanced": "Marketing-focused with quality indicators (12-18 words)",
    "seo_optimized": "Keyword-rich for search engines (15-25 words)"
  },
  "analysis": {
    "color": "primary color",
    "material": "material type",
    "product_type": "category",
    "style": "design style",
    "visible_features": ["feature1", "feature2"]
  }
}"""

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        }
                    ],
                },
            ],
            temperature=0.4,
            max_tokens=600,
        )

        raw = resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        
        if "captions" not in parsed or "analysis" not in parsed:
            raise ValueError("Invalid response structure")
        
        return parsed
        
    except Exception as e:
        basic = analyze_image(image_bytes)
        caption = basic.get("caption", "Product")
        
        return {
            "captions": {
                "standard": caption,
                "enhanced": f"{caption} | Premium Quality",
                "seo_optimized": f"{caption} - High Quality Product"
            },
            "analysis": basic,
            "error": str(e)
        }


def analyze_description(description: str, category: str) -> Dict[str, Any]:
    """Analyze product description using Chain-of-Thought reasoning."""
    description = sanitize_text_input(description, max_length=2000)
    category = sanitize_text_input(category, max_length=100)
    
    client = get_openai_client()

    system_prompt = """You are an NLP analyst specializing in e-commerce product descriptions.

Analyze the product description and return structured insights.

Return ONLY valid JSON:
{
  "keywords": ["extracted", "keywords"],
  "claims": ["quality claims made"],
  "implied_benefits": ["customer benefits"],
  "tone": "professional|casual|enthusiastic",
  "category_guess": "inferred category",
  "confidence": "high|medium|low"
}"""

    try:
        user_message = f"""Category: {category}

Description: {description}

Analyze this description."""

        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        raw = resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        
        defaults = {
            "keywords": [],
            "claims": [],
            "implied_benefits": [],
            "tone": "unknown",
            "category_guess": category,
            "confidence": "low"
        }
        
        for key, default_val in defaults.items():
            if key not in parsed:
                parsed[key] = default_val
        
        return parsed
        
    except Exception as e:
        return {
            "keywords": [],
            "claims": [],
            "implied_benefits": [],
            "tone": "unknown",
            "category_guess": category,
            "confidence": "low",
            "error": str(e)
        }


def semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using embeddings."""
    text1 = sanitize_text_input(text1, max_length=1000)
    text2 = sanitize_text_input(text2, max_length=1000)
    
    if not text1 or not text2:
        return 0.0
    
    client = get_openai_client()

    try:
        emb = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text1, text2],
        )

        vec_a = emb.data[0].embedding
        vec_b = emb.data[1].embedding

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))
        
        similarity = dot_product / (magnitude_a * magnitude_b + 1e-8)
        return max(0.0, min(1.0, similarity))
        
    except Exception:
        return 0.5


def compute_risk_score(similarity: float, missing_feats: List[str], contradictions: List[str]) -> int:
    """Calculate risk score based on multiple factors."""
    score = 0

    if similarity < 0.3:
        score += 40
    elif similarity < 0.5:
        score += 25
    elif similarity < 0.7:
        score += 10

    score += len(missing_feats) * 5
    score += len(contradictions) * 25

    return min(score, 100)


def generate_suggestions(
    image_analysis: Dict[str, Any],
    description: str,
    price: float,
    category: str,
    risk_score: int,
    similarity: float
) -> List[Dict[str, str]]:
    """Generate actionable improvement suggestions."""
    description = sanitize_text_input(description, max_length=1000)
    category = sanitize_text_input(category, max_length=100)
    
    client = get_openai_client()
    
    system_prompt = """You are an e-commerce optimization consultant.

Generate 3-7 improvement suggestions based on the product analysis.

Return ONLY valid JSON array:
[
  {
    "type": "missing_info|consistency|seo|marketing",
    "icon": "relevant emoji",
    "title": "Brief actionable title (5-8 words)",
    "description": "Specific action to take (20-40 words)",
    "priority": "critical|high|medium|low"
  }
]"""

    try:
        user_message = f"""Product Analysis:
- Image: {json.dumps(image_analysis)}
- Description: "{description}"
- Price: ${price}
- Category: {category}
- Risk Score: {risk_score}/100
- Similarity: {similarity:.2%}

Generate improvement suggestions."""

        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1000,
        )

        raw = resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        
        if isinstance(parsed, dict):
            parsed = [parsed]
        elif not isinstance(parsed, list):
            parsed = []
        
        return parsed[:10]
        
    except Exception as e:
        suggestions = []
        
        if not image_analysis.get("color"):
            suggestions.append({
                "type": "missing_info",
                "icon": "🎨",
                "title": "Add Color Information",
                "description": "Specify product color in description to improve searchability.",
                "priority": "medium"
            })
        
        if similarity < 0.6:
            suggestions.append({
                "type": "consistency",
                "icon": "📝",
                "title": "Improve Image-Description Match",
                "description": f"Description only {int(similarity*100)}% matches image. Add visible features.",
                "priority": "high"
            })
        
        return suggestions


def generate_improved_description(
    image_analysis: Dict[str, Any],
    original_description: str,
    price: float,
    category: str
) -> str:
    """Generate AI-improved product description."""
    original_description = sanitize_text_input(original_description, max_length=1000)
    category = sanitize_text_input(category, max_length=100)
    
    client = get_openai_client()
    
    system_prompt = """You are an expert e-commerce copywriter.

Rewrite the product description to be more effective and conversion-focused.

- Length: 60-100 words
- Include specific features
- Clear, benefit-oriented language
- Natural keyword placement

Return ONLY the improved description text (no JSON, no markdown)."""

    try:
        visible_features = image_analysis.get("visible_features", [])
        features_str = ", ".join(visible_features[:5]) if visible_features else "standard features"
        
        user_message = f"""Original: "{original_description}"

Image shows: {image_analysis.get('caption', 'N/A')}
Color: {image_analysis.get('color', 'N/A')}
Material: {image_analysis.get('material', 'N/A')}
Features: {features_str}
Category: {category}
Price: ${price}

Rewrite this description."""

        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=250,
        )

        improved = resp.choices[0].message.content.strip()
        
        if len(improved) < 20:
            return original_description
        
        return improved
        
    except Exception as e:
        return original_description


def generate_reviews_for_product(
    caption: str,
    description: str,
    price: float,
    category: str
) -> List[Dict[str, Any]]:
    """Generate simulated customer reviews."""
    client = get_openai_client()

    prompt = f"""Generate 4 realistic customer reviews for this product.
Return STRICT JSON ONLY as array:
[
  {{"rating": 5, "title": "...", "body": "..."}},
  {{"rating": 4, "title": "...", "body": "..."}},
  {{"rating": 3, "title": "...", "body": "..."}},
  {{"rating": 1, "title": "...", "body": "..."}}
]

Product: {caption}
Description: {description}
Category: {category}
Price: ${price}"""

    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.4,
        )
        
        raw = resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        
        if isinstance(parsed, list):
            return parsed
        return []
    except:
        return []


def full_analysis(
    image_bytes: bytes,
    description: str,
    price: float,
    category: str,
    generate_captions: bool = True
) -> Dict[str, Any]:
    """Complete product analysis pipeline."""
    description = sanitize_text_input(description)
    category = sanitize_text_input(category)
    
    if generate_captions:
        caption_result = generate_multiple_captions(image_bytes)
        image_analysis = caption_result.get("analysis", {})
        captions = caption_result.get("captions", {})
    else:
        image_analysis = analyze_image(image_bytes)
        captions = {"standard": image_analysis.get("caption", "")}
    
    description_analysis = analyze_description(description, category)
    
    caption = captions.get("standard", "")
    similarity = semantic_similarity(caption, description)
    
    missing = []
    if not image_analysis.get("color"):
        missing.append("color")
    if not image_analysis.get("material"):
        missing.append("material")
    if not image_analysis.get("product_type"):
        missing.append("product_type")
    
    contradictions = []
    risk = compute_risk_score(similarity, missing, contradictions)
    
    suggestions = generate_suggestions(
        image_analysis=image_analysis,
        description=description,
        price=price,
        category=category,
        risk_score=risk,
        similarity=similarity
    )
    
    comparison = {
        "img_caption": caption,
        "similarity": similarity,
        "missing_features": missing,
        "contradictions": contradictions,
        "risk_score": risk,
    }
    
    return {
        "image_analysis": image_analysis,
        "captions": captions,
        "description_analysis": description_analysis,
        "comparison": comparison,
        "suggestions": suggestions,
        "timestamp": datetime.utcnow().isoformat(),
    }


def parse_json_response(raw: str) -> Any:
    """Parse JSON from LLM response with fallback strategies."""
    try:
        return json.loads(raw)
    except:
        pass
    
    if "```" in raw:
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except:
            pass
    
    if "{" in raw and "}" in raw:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        try:
            return json.loads(raw[start:end])
        except:
            pass
    
    if "[" in raw and "]" in raw:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        try:
            return json.loads(raw[start:end])
        except:
            pass
    
    return {"error": "JSON parse failed", "raw": raw[:200]}


def build_report_text(
    image_analysis: Dict,
    description: str,
    comparison: Dict,
    reviews: List[Dict],
    price: float,
    category: str,
    captions: Dict = None,
    suggestions: List[Dict] = None,
) -> str:
    """Build comprehensive text report."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("SMART E-COMMERCE PRODUCT ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Product Info
    lines.append("PRODUCT INFORMATION")
    lines.append("-" * 70)
    lines.append(f"Category: {category}")
    lines.append(f"Price: ${price:.2f}")
    lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Original Description
    lines.append("ORIGINAL DESCRIPTION")
    lines.append("-" * 70)
    lines.append(description)
    lines.append("")
    
    # AI-Generated Captions
    if captions:
        lines.append("AI-GENERATED CAPTIONS")
        lines.append("-" * 70)
        lines.append(f"Standard: {captions.get('standard', 'N/A')}")
        lines.append(f"Enhanced: {captions.get('enhanced', 'N/A')}")
        lines.append(f"SEO Optimized: {captions.get('seo_optimized', 'N/A')}")
        lines.append("")
    
    # Image Analysis
    lines.append("IMAGE ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"Detected Color: {image_analysis.get('color', 'N/A')}")
    lines.append(f"Material: {image_analysis.get('material', 'N/A')}")
    lines.append(f"Product Type: {image_analysis.get('product_type', 'N/A')}")
    lines.append(f"Style: {image_analysis.get('style', 'N/A')}")
    
    features = image_analysis.get('visible_features', [])
    if features:
        lines.append(f"Visible Features: {', '.join(features)}")
    lines.append("")
    
    # Quality Metrics
    lines.append("QUALITY ASSESSMENT")
    lines.append("-" * 70)
    risk_score = comparison.get('risk_score', 0)
    similarity = comparison.get('similarity', 0)
    
    lines.append(f"Risk Score: {risk_score}/100")
    if risk_score < 30:
        lines.append("Risk Level: LOW - Excellent listing quality")
    elif risk_score < 60:
        lines.append("Risk Level: MEDIUM - Some improvements recommended")
    else:
        lines.append("Risk Level: HIGH - Significant issues detected")
    
    lines.append(f"Image-Description Similarity: {int(similarity*100)}%")
    
    missing = comparison.get('missing_features', [])
    if missing:
        lines.append(f"Missing Information: {', '.join(missing)}")
    lines.append("")
    
    # AI Recommendations
    if suggestions:
        lines.append("AI RECOMMENDATIONS")
        lines.append("-" * 70)
        for idx, sug in enumerate(suggestions, 1):
            priority = sug.get('priority', 'low').upper()
            title = sug.get('title', '')
            desc = sug.get('description', '')
            lines.append(f"{idx}. [{priority}] {title}")
            lines.append(f"   {desc}")
            lines.append("")
    
    # Customer Reviews (if available)
    if reviews:
        lines.append("SIMULATED CUSTOMER REVIEWS")
        lines.append("-" * 70)
        for idx, review in enumerate(sorted(reviews, key=lambda r: r.get('rating', 0), reverse=True), 1):
            rating = review.get('rating', 0)
            title = review.get('title', '')
            body = review.get('body', '')
            stars = "★" * rating + "☆" * (5 - rating)
            
            lines.append(f"Review {idx}: {stars} ({rating}/5)")
            lines.append(f"Title: {title}")
            lines.append(f"Body: {body}")
            lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 70)
    if risk_score < 30:
        lines.append("✓ This listing is well-optimized and ready for publication.")
    elif risk_score < 60:
        lines.append("! This listing has some areas for improvement. Review recommendations above.")
    else:
        lines.append("✗ This listing needs significant improvements before publication.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Smart E-Commerce Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main {padding: 1rem 2rem;}
    h1 {color: #1e3a8a; margin-bottom: 0.5rem;}
    h2 {color: #1e40af; margin-top: 2rem; margin-bottom: 1rem;}
    h3 {color: #3b82f6; margin-top: 1rem;}
    
    .caption-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid;
        margin: 1rem 0;
    }
    
    .risk-low {background: #d1fae5; border-color: #10b981;}
    .risk-medium {background: #fef3c7; border-color: #f59e0b;}
    .risk-high {background: #fee2e2; border-color: #ef4444;}
    
    .suggestion {
        background: white;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    
    .sug-critical {border-color: #dc2626; background: #fef2f2;}
    .sug-high {border-color: #f97316; background: #fff7ed;}
    .sug-medium {border-color: #eab308; background: #fefce8;}
    .sug-low {border-color: #3b82f6; background: #eff6ff;}
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in {
    "analysis": None,
    "image_bytes": None,
    "description": "",
    "price": 19.99,
    "category": "Furniture",
    "reviews": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Header
st.title("🛒 Smart E-Commerce Assistant")
st.caption("AI-powered product listing analyzer with intelligent recommendations")

# Quick stats
if st.session_state.analysis:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    comparison = st.session_state.analysis.get("comparison", {})
    risk_score = comparison.get("risk_score", 0)
    similarity = comparison.get("similarity", 0)
    
    with col1:
        st.metric("Risk Score", f"{risk_score}/100")
    with col2:
        st.metric("Match Quality", f"{int(similarity*100)}%")
    with col3:
        st.metric("Reviews", len(st.session_state.reviews) if st.session_state.reviews else 0)
    with col4:
        if st.session_state.reviews:
            avg = sum(r.get("rating", 0) for r in st.session_state.reviews) / len(st.session_state.reviews)
            st.metric("Avg Rating", f"{avg:.1f}★")

st.markdown("---")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📸 Product Analysis", "⭐ Customer Reviews", "📊 Full Report"])

# ========== TAB 1: ANALYSIS ==========
with tab1:
    left_col, right_col = st.columns([1, 1.4], gap="large")
    
    with left_col:
        st.markdown("## 📥 Input")
        
        uploaded_file = st.file_uploader("**Product Image**", type=["png", "jpg", "jpeg", "webp"])
        
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
        
        st.markdown("---")
        
        description = st.text_area(
            "**Product Description**",
            value=st.session_state.description,
            height=150,
            placeholder="Enter your product description..."
        )
        
        col_p, col_c = st.columns(2)
        with col_p:
            price = st.number_input("**Price ($)**", min_value=0.0, value=float(st.session_state.price), step=1.0)
        with col_c:
            category = st.selectbox(
                "**Category**",
                ["Phone Case", "Furniture", "Clothing", "Electronics", "Home Decor", "Toys", "Sports", "Other"],
                index=1
            )
        
        st.markdown("")
        analyze_btn = st.button("🔍 **Analyze Product**", type="primary", use_container_width=True)
        
        if analyze_btn:
            if not uploaded_file:
                st.error("📷 Upload an image")
            elif not description.strip() or len(description.strip()) < 10:
                st.error("📝 Enter description (min 10 chars)")
            elif price <= 0:
                st.error("💰 Enter valid price")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        image_bytes = uploaded_file.getvalue()
                        
                        analysis = full_analysis(
                            image_bytes=image_bytes,
                            description=description,
                            price=price,
                            category=category,
                            generate_captions=True
                        )
                        
                        st.session_state.analysis = analysis
                        st.session_state.image_bytes = image_bytes
                        st.session_state.description = description
                        st.session_state.price = price
                        st.session_state.category = category
                        
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
    
    with right_col:
        st.markdown("## 📤 AI Analysis Results")
        
        if not st.session_state.analysis:
            st.info("👈 **Upload a product image and click Analyze**")
        else:
            analysis = st.session_state.analysis
            captions = analysis.get("captions", {})
            comparison = analysis.get("comparison", {})
            suggestions = analysis.get("suggestions", [])
            
            # Captions
            st.markdown("### 🎨 AI-Generated Captions")
            
            for cap_type, badge_class, label in [
                ("standard", "badge-standard", "STANDARD - Professional"),
                ("enhanced", "badge-enhanced", "ENHANCED - Marketing"),
                ("seo_optimized", "badge-seo", "SEO - Keyword-Rich")
            ]:
                caption_text = captions.get(cap_type, "N/A")
                st.markdown(f"""
                <div class="caption-card">
                    <div style="font-weight: 700; margin-bottom: 0.5rem;">{label}</div>
                    <div style="color: #374151;">{caption_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"📋 Copy", key=f"copy_{cap_type}"):
                    st.code(caption_text)
            
            st.markdown("---")
            
            # Risk Assessment
            st.markdown("### 🎯 Risk Assessment")
            
            risk_score = comparison.get("risk_score", 0)
            similarity = comparison.get("similarity", 0)
            
            if risk_score < 30:
                risk_class = "risk-low"
                risk_label = "🟢 LOW RISK"
                risk_msg = "✅ Excellent! Image and description align well."
            elif risk_score < 60:
                risk_class = "risk-medium"
                risk_label = "🟡 MEDIUM RISK"
                risk_msg = "⚠️ Some issues detected. Review recommended."
            else:
                risk_class = "risk-high"
                risk_label = "🔴 HIGH RISK"
                risk_msg = "❌ Major mismatches found. Update required."
            
            st.markdown(f"""
            <div class="risk-box {risk_class}">
                <h3 style="margin: 0 0 0.5rem 0;">{risk_label}</h3>
                <p style="margin: 0;">{risk_msg}</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Risk: <strong>{risk_score}/100</strong> | 
                    Similarity: <strong>{int(similarity*100)}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Suggestions
            if suggestions:
                st.markdown("### 💡 AI Recommendations")
                
                for sug in suggestions:
                    priority = sug.get("priority", "low")
                    icon = sug.get("icon", "💡")
                    title = sug.get("title", "")
                    desc = sug.get("description", "")
                    
                    st.markdown(f"""
                    <div class="suggestion sug-{priority}">
                        <strong>{icon} {title}</strong>
                        <span style="font-size: 0.75rem; color: #6b7280;">[{priority.upper()}]</span>
                        <p style="margin: 0.5rem 0 0 0; color: #4b5563;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Improved Description
            with st.expander("✨ Generate AI-Improved Description"):
                if st.button("🤖 Generate Improved Description"):
                    with st.spinner("Generating..."):
                        try:
                            improved = generate_improved_description(
                                image_analysis=analysis["image_analysis"],
                                original_description=st.session_state.description,
                                price=st.session_state.price,
                                category=st.session_state.category
                            )
                            
                            st.markdown("**🔵 Original:**")
                            st.info(st.session_state.description)
                            
                            st.markdown("**🟢 AI-Improved:**")
                            st.success(improved)
                        except Exception as e:
                            st.error(f"Error: {e}")

# ========== TAB 2: REVIEWS ==========
with tab2:
    st.markdown("## ⭐ Customer Review Simulation")
    
    if not st.session_state.analysis:
        st.info("👈 Complete product analysis first")
    else:
        if st.button("🎲 Generate Reviews", type="primary"):
            with st.spinner("Generating..."):
                try:
                    captions = st.session_state.analysis.get("captions", {})
                    caption = captions.get("standard", "")
                    
                    reviews = generate_reviews_for_product(
                        caption=caption,
                        description=st.session_state.description,
                        price=st.session_state.price,
                        category=st.session_state.category
                    )
                    
                    st.session_state.reviews = reviews
                    st.success("✅ Reviews generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.reviews:
            st.markdown("---")
            
            for review in sorted(st.session_state.reviews, key=lambda r: r.get("rating", 0), reverse=True):
                rating = review.get("rating", 0)
                title = review.get("title", "")
                body = review.get("body", "")
                stars = "⭐" * rating + "☆" * (5 - rating)
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{stars}** {title}")
                with col2:
                    st.caption(f"{rating}/5 ⭐")
                
                st.markdown(f"_{body}_")
                st.caption("👤 Verified Purchase")
                st.markdown("---")

# ========== TAB 3: FULL REPORT ==========
with tab3:
    st.markdown("## 📊 Comprehensive Report")
    
    if not st.session_state.analysis:
        st.info("👈 Complete product analysis first")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📄 Generate Full Report")
            st.caption("Combines all analysis results into a downloadable report")
        
        with col2:
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Creating report..."):
                    try:
                        analysis = st.session_state.analysis
                        captions = analysis.get("captions", {})
                        comparison = analysis.get("comparison", {})
                        suggestions = analysis.get("suggestions", [])
                        
                        # Generate reviews if not already done
                        if not st.session_state.reviews:
                            caption = captions.get("standard", "")
                            reviews = generate_reviews_for_product(
                                caption=caption,
                                description=st.session_state.description,
                                price=st.session_state.price,
                                category=st.session_state.category
                            )
                            st.session_state.reviews = reviews
                        
                        report_text = build_report_text(
                            image_analysis=analysis["image_analysis"],
                            description=st.session_state.description,
                            comparison=comparison,
                            reviews=st.session_state.reviews,
                            price=st.session_state.price,
                            category=st.session_state.category,
                            captions=captions,
                            suggestions=suggestions
                        )
                        
                        st.session_state["report_text"] = report_text
                        st.success("✅ Report generated!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Display report if generated
        if st.session_state.get("report_text"):
            st.markdown("---")
            
            # Report preview
            with st.expander("📄 View Text Report", expanded=True):
                st.text(st.session_state["report_text"])
            
            # Download options
            st.markdown("### 💾 Download Options")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                st.download_button(
                    "📥 Download as TXT",
                    data=st.session_state["report_text"],
                    file_name="product_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_d2:
                # Create PDF
                if st.button("📥 Generate PDF", use_container_width=True):
                    with st.spinner("Creating PDF..."):
                        try:
                            # Simple PDF generation
                            buffer = io.BytesIO()
                            c = canvas.Canvas(buffer, pagesize=A4)
                            width, height = A4
                            
                            # Add title
                            c.setFont("Helvetica-Bold", 16)
                            c.drawString(40, height - 40, "Product Analysis Report")
                            
                            # Add report text
                            c.setFont("Helvetica", 10)
                            y = height - 80
                            for line in st.session_state["report_text"].split("\n")[:60]:
                                if y < 50:
                                    c.showPage()
                                    y = height - 40
                                c.drawString(40, y, line[:90])
                                y -= 14
                            
                            c.save()
                            pdf_bytes = buffer.getvalue()
                            
                            st.download_button(
                                "⬇️ Download PDF",
                                data=pdf_bytes,
                                file_name="product_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"PDF Error: {e}")
            
            with col_d3:
                # Copy to clipboard
                if st.button("📋 Copy Report", use_container_width=True):
                    st.code(st.session_state["report_text"])

st.caption("🤖 Powered by GPT-4 Vision • Smart E-Commerce Assistant")