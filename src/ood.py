import re

from .data import SLOTS

WORD_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+")

NOUNS = set(SLOTS["producto"] + SLOTS["lugar"] + SLOTS["situacion"])
NOUNS.update({
    "producto", "servicio", "artículo", "articulo", "sucursal", "tienda", "web",
    "catálogo", "catalogo", "app", "almacén", "almacen", "mostrador", "soporte",
    "atención", "atencion", "cliente", "empresa", "factura", "envío", "envio",
    "pedido", "paquete", "cuenta", "garantía", "garantia", "devolución", "devolucion",
})

ACTIONS = set(SLOTS["info"] + SLOTS["accion_consulta"])
ACTIONS.update({
    "queja", "consulta", "problema", "duda", "información", "informacion", "precio",
    "cuesta", "costo", "dañado", "daniado", "fallas", "tardío", "tardio", "funciona",
    "cobro", "cargo", "reembolso", "respuesta", "resolver", "solución", "solucion",
    "ayuda", "esperar", "recibir", "entrega", "pago", "pagar", "horario", "plazo",
    "detalle", "seguir", "esperando", "llegó", "llego", "envío", "envio",
})

STOPWORDS = {
    "el", "la", "los", "las", "mi", "mis", "un", "una", "de", "del", "al", "en",
    "con", "por", "para", "que", "se", "su", "sus", "y", "o", "a", "si", "no",
    "cómo", "como", "donde", "cuál", "cual", "cuánto", "cuanto", "qué", "que",
    "podrían", "poderian", "puedo", "quiero", "necesito", "me", "gustaría", "gustaria",
    "sin", "desde", "todavía", "todavia", "más", "mas", "hay", "es", "está", "esta",
}

for _key in ["producto", "producto_det", "lugar", "situacion", "servicio", "medio"]:
    for _value in SLOTS[_key]:
        NOUNS.update(w for w in WORD_RE.findall(_value.lower()) if w not in STOPWORDS)
for _key in ["info", "accion_consulta", "estado", "accion", "funcion", "respuesta", "problema_cobro"]:
    for _value in SLOTS[_key]:
        ACTIONS.update(w for w in WORD_RE.findall(_value.lower()) if w not in STOPWORDS)

ENGLISH_MARKERS = re.compile(
    r"\b(the|and|damaged|refund|arrived|please|package|item|hello|weather|want|with|your)\b",
    re.IGNORECASE,
)
PERSONAL_PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
PERSONAL_NAME = re.compile(r"\b(señor|señora|sr|sra)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def normalize(text):
    cleaned = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+", " ", str(text).lower())
    return " ".join(cleaned.split())


def ood_reason(text):
    raw = str(text)
    if not raw.strip():
        return "entrada vacía o solo espacios"
    words = WORD_RE.findall(raw)
    if len(words) < 3:
        return "texto demasiado corto o sin contenido alfabético suficiente"
    if EMAIL.search(raw) or PERSONAL_PHONE.search(raw) or PERSONAL_NAME.search(raw):
        return "posible dato personal: se rechaza y no se almacena"
    if ENGLISH_MARKERS.search(raw):
        return "texto en otro idioma"
    normalized = normalize(raw)
    tokens = set(normalized.split())
    if not (tokens & NOUNS):
        return "sin vocabulario del dominio de atención a cliente"
    if not (tokens & ACTIONS):
        return "sin acción o problema reconocible del dominio"
    return None
