from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
import uuid
import time

# Create a custom logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Error response model for consistent error reporting
class ErrorResponse(BaseModel):
    status_code: int
    error: str
    detail: Any
    path: str
    timestamp: str
    request_id: str

# Request tracking middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log the incoming request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Track request time
    start_time = time.time()
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Log the response time and status
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.3f}s")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        # Log unhandled exceptions
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a formatted error response
        error_response = ErrorResponse(
            status_code=500,
            error="Internal Server Error",
            detail=str(e),
            path=request.url.path,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )

# Handler for Pydantic validation errors (422 Unprocessable Entity)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors from FastAPI/Pydantic with detailed field errors
    """
    # Extract request ID
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Format validation errors into a more readable structure
    formatted_errors = []
    for error in exc.errors():
        # Extract the field location, type, and message
        loc = " -> ".join(str(loc_item) for loc_item in error.get("loc", []))
        err_type = error.get("type", "")
        msg = error.get("msg", "")
        
        formatted_errors.append({
            "field": loc,
            "type": err_type,
            "message": msg
        })
    
    # Log validation error with useful context
    logger.warning(f"Validation error on request {request_id} to {request.url.path}: {formatted_errors}")
    
    # Create error response
    error_response = ErrorResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error="Validation Error",
        detail={
            "errors": formatted_errors,
            "body": await get_request_body(request)  # Add the request body for debugging
        },
        path=request.url.path,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )

# Handler for JSON decode errors
@app.exception_handler(json.JSONDecodeError)
async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
    """
    Handle malformed JSON in request body
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning(f"JSON decode error on request {request_id}: {str(exc)}")
    
    # Try to get the raw body to include in the error response
    try:
        body = await request.body()
        body_str = body.decode("utf-8")
    except Exception:
        body_str = "<Failed to decode request body>"
    
    error_response = ErrorResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        error="Invalid JSON",
        detail={
            "message": f"Invalid JSON format at position {exc.pos}: {exc.msg}",
            "received_body": body_str[:1000]  # Limit to first 1000 chars
        },
        path=request.url.path,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump()
    )

# Handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with standard format
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning(f"HTTP {exc.status_code} error on request {request_id}: {exc.detail}")
    
    error_response = ErrorResponse(
        status_code=exc.status_code,
        error=get_error_title(exc.status_code),
        detail=exc.detail,
        path=request.url.path,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content=error_response.model_dump()
    )

# General exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unhandled exceptions
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"Unhandled exception on request {request_id}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    error_response = ErrorResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error="Internal Server Error",
        detail="An unexpected error occurred. Please try again later.",
        path=request.url.path,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )

# Utility function to safely get request body
async def get_request_body(request: Request) -> Dict:
    """
    Get request body as JSON or string, handling potential errors
    """
    try:
        return await request.json()
    except json.JSONDecodeError:
        try:
            body = await request.body()
            return {"raw": body.decode("utf-8")}
        except:
            return {"raw": "<Failed to decode body>"}

# Utility function to map status codes to error titles
def get_error_title(status_code: int) -> str:
    """
    Map status codes to human-readable titles
    """
    status_titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        409: "Conflict",
        413: "Request Entity Too Large",
        415: "Unsupported Media Type",
        422: "Validation Error",
        429: "Too Many Requests",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable"
    }
    return status_titles.get(status_code, "Error")

# Sample endpoint that shows how to properly handle exceptions
@app.post("/analyze_profile", response_model=UserCategorization)
async def analyze_profile(request: Request, user_profile: UserProfile):
    """
    Analyze user investment profile and recommend stocks
    """
    # Get request ID for tracking
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Processing profile analysis (ID: {request_id})")
    
    try:
        # Log the request data at debug level
        logger.debug(f"Profile data: {user_profile.model_dump()}")
        
        # Validate assets indices
        for asset_idx in user_profile.assets:
            if asset_idx < 0 or asset_idx >= len(asset_options):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid asset index: {asset_idx}. Must be between 0 and {len(asset_options)-1}"
                )
        
        # Convert user profile to vector format
        assets_vector = [1 if i in user_profile.assets else 0 for i in range(len(asset_options))]
        user_vector = [
            user_profile.goal,
            user_profile.risk,
            user_profile.frequency,
            *assets_vector,
            user_profile.volatility_reaction,
            user_profile.time_horizon,
            user_profile.decision_making,
            user_profile.emotion,
            user_profile.capital,
            user_profile.trading_style
        ]
        
        # Categorize user
        category = categorize_user(user_vector)
        logger.info(f"User categorized as: {category}")
        
        # Get stock recommendations
        stocks = recommend_stocks(category)
        
        response = UserCategorization(
            category=category,
            recommendations=stocks
        )
        
        return response
    except HTTPException as exc:
        # Re-raise HTTP exceptions to be handled by the HTTP exception handler
        raise exc
    except Exception as exc:
        # Log unexpected errors
        logger.error(f"Unexpected error in analyze_profile: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Raise as HTTPException to be handled by the HTTP exception handler
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze profile: {str(exc)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    # Simple health check - could be expanded to check external dependencies
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_version": "1.0.0"
    }

# Debug endpoint to test validation errors (handy for troubleshooting)
@app.get("/test-validation")
async def test_validation():
    """
    Test endpoint that shows how validation errors are handled
    """
    # Simulate validation error by raising a ValidationError
    data = {"invalid_key": "value"}
    raise RequestValidationError([{
        "loc": ("body", "user_profile", "assets"),
        "msg": "field required",
        "type": "value_error.missing"
    }])

# Debug endpoint to test JSON errors
@app.post("/test-json")
async def test_json(request: Request):
    """
    Test endpoint that helps debug JSON parsing issues
    """
    try:
        # Attempt to parse JSON
        data = await request.json()
        return {"received": data}
    except json.JSONDecodeError as e:
        # This should be caught by the exception handler
        raise e
