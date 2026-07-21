"""
Domain errors.

Every failure the pipeline can produce maps to a stable machine-readable
code, an HTTP status, and a user-friendly message. The API layer converts
these into a consistent JSON error envelope while full stack traces go to
the logs only.
"""


class AppError(Exception):
    """Base class for all expected, user-facing failures."""

    code = "INTERNAL_ERROR"
    http_status = 500
    user_message = "Something went wrong while processing your request. Please try again."

    def __init__(self, detail: str | None = None, user_message: str | None = None):
        # `detail` is for logs; `user_message` is what the client sees.
        self.detail = detail or self.user_message
        if user_message:
            self.user_message = user_message
        super().__init__(self.detail)


class MissingImageError(AppError):
    code = "MISSING_IMAGE"
    http_status = 400
    user_message = "Both the front and back images of the ID card are required."


class UnsupportedFormatError(AppError):
    code = "UNSUPPORTED_FORMAT"
    http_status = 415
    user_message = "Unsupported file type. Please upload a JPG, JPEG, PNG or WEBP image."


class FileTooLargeError(AppError):
    code = "FILE_TOO_LARGE"
    http_status = 413
    user_message = "The uploaded file is too large."


class CorruptedImageError(AppError):
    code = "CORRUPTED_IMAGE"
    http_status = 422
    user_message = "The uploaded file could not be read as an image. It may be corrupted."


class LowResolutionError(AppError):
    code = "LOW_RESOLUTION"
    http_status = 422
    user_message = (
        "The image resolution is too low for reliable scanning. "
        "Please capture a closer, higher-resolution photo."
    )


class BlurryImageError(AppError):
    code = "BLURRY_IMAGE"
    http_status = 422
    user_message = "The image appears too blurry to read. Hold the camera steady and retake the photo."


class CardNotDetectedError(AppError):
    code = "CARD_NOT_DETECTED"
    http_status = 422
    user_message = (
        "No ID card could be detected in the image. Place the card on a contrasting "
        "background, make sure all four corners are visible, and try again."
    )


class PartialCardError(AppError):
    code = "PARTIAL_CARD"
    http_status = 422
    user_message = "The ID card appears to be partially cut off. Make sure the whole card is in frame."


class OcrFailureError(AppError):
    code = "OCR_FAILURE"
    http_status = 502
    user_message = "Text could not be extracted from the card. Please retake the photo with better lighting."
