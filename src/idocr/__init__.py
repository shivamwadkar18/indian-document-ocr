"""idocr — OCR and document understanding for Indian identity documents.

Sub-packages
------------
data        dataset discovery, audit, conversion, preprocessing, augmentation, synthetic data
models      model interfaces: document classifier, text detector, text recognizer
training    training entry points (placeholders)
extraction  document-specific field extraction (Aadhaar, PAN, Driving Licence)
validation  field validation
evaluation  detection / recognition / end-to-end metrics
pipeline    end-to-end inference pipeline
utils       paths, config, image I/O, logging
"""

__version__ = "0.1.0"
