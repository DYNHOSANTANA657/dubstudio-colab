"""Catálogo local de idiomas e predefinições da interface."""

LANGUAGES = [
    ("af-ZA", "Africâner", "África"),
    ("sq-AL", "Albanês", "Europa"),
    ("am-ET", "Amárico", "África"),
    ("ar-SA", "Árabe", "Oriente Médio"),
    ("hy-AM", "Armênio", "Europa e Ásia"),
    ("az-AZ", "Azerbaijano", "Europa e Ásia"),
    ("eu-ES", "Basco", "Europa"),
    ("bn-BD", "Bengali", "Ásia"),
    ("bs-BA", "Bósnio", "Europa"),
    ("bg-BG", "Búlgaro", "Europa"),
    ("my-MM", "Birmanês", "Ásia"),
    ("ca-ES", "Catalão", "Europa"),
    ("zh-CN", "Chinês — mandarim simplificado", "Ásia"),
    ("zh-TW", "Chinês — mandarim tradicional", "Ásia"),
    ("hr-HR", "Croata", "Europa"),
    ("cs-CZ", "Tcheco", "Europa"),
    ("da-DK", "Dinamarquês", "Europa"),
    ("nl-NL", "Holandês", "Europa"),
    ("en-AU", "Inglês — Austrália", "Inglês"),
    ("en-CA", "Inglês — Canadá", "Inglês"),
    ("en-IN", "Inglês — Índia", "Inglês"),
    ("en-GB", "Inglês — Reino Unido", "Inglês"),
    ("en-US", "Inglês — Estados Unidos", "Inglês"),
    ("et-EE", "Estoniano", "Europa"),
    ("fil-PH", "Filipino", "Ásia"),
    ("fi-FI", "Finlandês", "Europa"),
    ("fr-BE", "Francês — Bélgica", "Francês"),
    ("fr-CA", "Francês — Canadá", "Francês"),
    ("fr-FR", "Francês — França", "Francês"),
    ("fr-CH", "Francês — Suíça", "Francês"),
    ("gl-ES", "Galego", "Europa"),
    ("ka-GE", "Georgiano", "Europa e Ásia"),
    ("de-AT", "Alemão — Áustria", "Alemão"),
    ("de-DE", "Alemão — Alemanha", "Alemão"),
    ("de-CH", "Alemão — Suíça", "Alemão"),
    ("el-GR", "Grego", "Europa"),
    ("gu-IN", "Gujarati", "Ásia"),
    ("he-IL", "Hebraico", "Oriente Médio"),
    ("hi-IN", "Hindi", "Ásia"),
    ("hu-HU", "Húngaro", "Europa"),
    ("is-IS", "Islandês", "Europa"),
    ("id-ID", "Indonésio", "Ásia"),
    ("ga-IE", "Irlandês", "Europa"),
    ("it-IT", "Italiano", "Europa"),
    ("ja-JP", "Japonês", "Ásia"),
    ("jv-ID", "Javanês", "Ásia"),
    ("kn-IN", "Canarês", "Ásia"),
    ("kk-KZ", "Cazaque", "Europa e Ásia"),
    ("km-KH", "Khmer", "Ásia"),
    ("ko-KR", "Coreano", "Ásia"),
    ("lo-LA", "Laosiano", "Ásia"),
    ("lv-LV", "Letão", "Europa"),
    ("lt-LT", "Lituano", "Europa"),
    ("mk-MK", "Macedônio", "Europa"),
    ("ms-MY", "Malaio", "Ásia"),
    ("ml-IN", "Malaiala", "Ásia"),
    ("mt-MT", "Maltês", "Europa"),
    ("mr-IN", "Marathi", "Ásia"),
    ("mn-MN", "Mongol", "Ásia"),
    ("ne-NP", "Nepalês", "Ásia"),
    ("nb-NO", "Norueguês", "Europa"),
    ("ps-AF", "Pashto", "Ásia"),
    ("fa-IR", "Persa", "Oriente Médio"),
    ("pl-PL", "Polonês — Polônia", "Europa"),
    ("pt-BR", "Português — Brasil", "Português"),
    ("pt-PT", "Português — Portugal", "Português"),
    ("pa-IN", "Punjabi", "Ásia"),
    ("ro-RO", "Romeno", "Europa"),
    ("ru-RU", "Russo", "Europa e Ásia"),
    ("sr-RS", "Sérvio", "Europa"),
    ("si-LK", "Cingalês", "Ásia"),
    ("sk-SK", "Eslovaco", "Europa"),
    ("sl-SI", "Esloveno", "Europa"),
    ("so-SO", "Somali", "África"),
    ("es-AR", "Espanhol — Argentina", "Espanhol"),
    ("es-CO", "Espanhol — Colômbia", "Espanhol"),
    ("es-ES", "Espanhol — Espanha", "Espanhol"),
    ("es-MX", "Espanhol — México", "Espanhol"),
    ("es-US", "Espanhol — Estados Unidos", "Espanhol"),
    ("sw-KE", "Suaíli", "África"),
    ("sv-SE", "Sueco", "Europa"),
    ("ta-IN", "Tâmil", "Ásia"),
    ("te-IN", "Telugu", "Ásia"),
    ("th-TH", "Tailandês", "Ásia"),
    ("tr-TR", "Turco", "Europa e Ásia"),
    ("uk-UA", "Ucraniano", "Europa"),
    ("ur-PK", "Urdu", "Ásia"),
    ("uz-UZ", "Uzbeque", "Ásia"),
    ("vi-VN", "Vietnamita", "Ásia"),
    ("cy-GB", "Galês", "Europa"),
    ("zu-ZA", "Zulu", "África"),
]

PRESETS = [
    {
        "id": "natural",
        "name": "Natural equilibrado",
        "description": "Mantém pausas, intenção e energia sem exageros.",
        "pace": 1.0,
        "expressiveness": 55,
        "mix": "broadcast",
    },
    {
        "id": "social",
        "name": "Social dinâmico",
        "description": "Ritmo ágil e presença para vídeos curtos.",
        "pace": 1.08,
        "expressiveness": 76,
        "mix": "presence",
    },
    {
        "id": "documentary",
        "name": "Documentário",
        "description": "Dicção firme, pausas naturais e menor variação.",
        "pace": 0.94,
        "expressiveness": 38,
        "mix": "warm",
    },
    {
        "id": "corporate",
        "name": "Corporativo claro",
        "description": "Voz limpa e neutra, otimizada para explicações.",
        "pace": 0.98,
        "expressiveness": 44,
        "mix": "clean",
    },
    {
        "id": "accessible",
        "name": "Leitura acessível",
        "description": "Velocidade reduzida e articulação reforçada.",
        "pace": 0.88,
        "expressiveness": 32,
        "mix": "dialogue",
    },
]


# Published model capability, not a claim that every language has passed a GPU run.
# Source: resemble-ai/chatterbox, src/chatterbox/mtl_tts.py (SUPPORTED_LANGUAGES).
CHATTERBOX_LANGUAGES = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
})
LANGUAGE_CAPABILITY_SOURCE = "https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/mtl_tts.py"


def model_language(code: str) -> str:
    base = str(code).lower().split("-")[0]
    return "no" if base == "nb" else base


def supports_original_voice(code: str) -> bool:
    return model_language(code) in CHATTERBOX_LANGUAGES


def languages_payload():
    payload = []
    for code, name, group in LANGUAGES:
        supported = supports_original_voice(code)
        payload.append({
            "code": code, "name": name, "group": group,
            "model_language": model_language(code),
            "base_language": model_language(code), "engine": "Chatterbox Multilingual",
            "supported": supported, "voice_clone": supported, "clone_supported": supported,
            "translation": True,
            "regional_accent_guaranteed": False,
            "regional_accent_verified": False,
            "validation_status": "pending_gpu_validation" if supported else "unsupported_voice_clone",
            "readiness": "pending_gpu_validation" if supported else "unsupported_voice_clone",
            "capability_source": LANGUAGE_CAPABILITY_SOURCE,
            "reason": "Validação em GPU pendente." if supported else "O motor atual não oferece timbre original neste idioma.",
        })
    return payload

