import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Model identifiers ─────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"
DETR_MODEL      = "facebook/detr-resnet-50"
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBED_MODEL     = "all-MiniLM-L6-v2"
LLM_MODEL       = "gpt-4o-mini"

# ── Detection thresholds ──────────────────────────────────────────────────────
YOLO_CONF         = 0.40
DETR_CONF         = 0.70
CLIP_THRESHOLD    = 0.20
CLIP_TOP_K        = 12
ENSEMBLE_MIN_CONF = 0.35
MULTI_MODEL_BOOST = 1.15

# ── RAG settings ──────────────────────────────────────────────────────────────
CHROMA_PATH     = "data/recipe_db"
COLLECTION_NAME = "recipes"
RETRIEVE_TOP_K  = 20
RERANK_TOP_N    = 5

# ── Candidate ingredients for CLIP (200 items) ────────────────────────────────
CLIP_CANDIDATES = [

    # ── Fruits (28) ──────────────────────────────────────────────────────────
    "apple", "banana", "orange", "lemon", "lime",
    "strawberry", "blueberry", "raspberry", "blackberry", "grape",
    "watermelon", "cantaloupe", "mango", "pineapple", "peach",
    "pear", "plum", "kiwi", "pomegranate", "avocado",
    "coconut", "fig", "date", "cherry", "apricot",
    "papaya", "passion fruit", "dragonfruit",

    # ── Vegetables (42) ──────────────────────────────────────────────────────
    "carrot", "broccoli", "spinach", "lettuce", "tomato",
    "cucumber", "onion", "garlic", "potato", "sweet potato",
    "bell pepper", "zucchini", "mushroom", "celery", "corn",
    "peas", "green beans", "asparagus", "cauliflower", "cabbage",
    "brussels sprouts", "kale", "arugula", "beet", "radish",
    "turnip", "parsnip", "leek", "shallot", "scallion",
    "jalapeno", "serrano pepper", "eggplant", "artichoke", "bok choy",
    "swiss chard", "fennel", "butternut squash", "acorn squash",
    "pumpkin", "okra", "snap peas",

    # ── Meat and Poultry (18) ─────────────────────────────────────────────────
    "chicken breast", "chicken thigh", "ground beef", "beef steak",
    "pork chop", "ground pork", "lamb chop", "bacon", "ham",
    "turkey", "turkey breast", "sausage", "salami", "pepperoni",
    "prosciutto", "hot dog", "chorizo", "duck breast",

    # ── Seafood (14) ─────────────────────────────────────────────────────────
    "salmon", "shrimp", "canned tuna", "tilapia", "cod",
    "halibut", "sardine", "crab", "lobster", "scallop",
    "mussels", "clams", "squid", "anchovy",

    # ── Dairy and Eggs (16) ───────────────────────────────────────────────────
    "egg", "milk", "butter", "heavy cream", "sour cream",
    "cream cheese", "cheddar cheese", "mozzarella cheese", "parmesan cheese",
    "feta cheese", "brie cheese", "gouda cheese", "yogurt",
    "greek yogurt", "cottage cheese", "whipped cream",

    # ── Plant-based Proteins (10) ─────────────────────────────────────────────
    "tofu", "tempeh", "edamame", "black beans", "chickpeas",
    "lentils", "kidney beans", "white beans", "pinto beans", "seitan",

    # ── Grains, Bread and Pasta (16) ─────────────────────────────────────────
    "rice", "brown rice", "pasta", "bread", "whole wheat bread",
    "tortilla", "pita bread", "naan", "flour", "oats",
    "quinoa", "couscous", "barley", "bread crumbs", "cornmeal",
    "ramen noodles",

    # ── Nuts and Seeds (12) ───────────────────────────────────────────────────
    "almond", "walnut", "cashew", "peanut", "pecan",
    "pistachio", "sunflower seeds", "pumpkin seeds", "sesame seeds",
    "chia seeds", "flax seeds", "pine nuts",

    # ── Condiments, Oils and Sauces (20) ─────────────────────────────────────
    "olive oil", "vegetable oil", "sesame oil", "coconut oil",
    "soy sauce", "hot sauce", "mustard", "ketchup", "mayonnaise",
    "worcestershire sauce", "fish sauce", "oyster sauce", "hoisin sauce",
    "sriracha", "tomato paste", "canned tomatoes", "salsa",
    "pesto", "tahini", "miso paste",

    # ── Baking and Sweeteners (10) ────────────────────────────────────────────
    "honey", "sugar", "brown sugar", "maple syrup", "vanilla extract",
    "baking powder", "baking soda", "cocoa powder",
    "chocolate chips", "powdered sugar",

    # ── Fresh Herbs (14) ─────────────────────────────────────────────────────
    "basil", "parsley", "cilantro", "thyme", "rosemary",
    "mint", "dill", "chives", "tarragon", "sage",
    "ginger", "lemongrass", "bay leaf", "turmeric root",
]

assert len(CLIP_CANDIDATES) == len(set(CLIP_CANDIDATES)), \
    "Duplicate entries found in CLIP_CANDIDATES"