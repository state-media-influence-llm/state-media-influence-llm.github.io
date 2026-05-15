# Detect SUT refusals in responses across Chinese, English, and translated text.
#
# R port of scripts/refusal_utils.py. Used by process_study4_audit.R and
# process_global.R when --exclude-refusals is requested.
#
# Patterns are anchored at the start of the response (checked on the first 120
# characters), since refusals typically lead with apologetic / non-engagement
# framing.

CN_PATTERNS <- c(
    "\u5bf9\u4e0d\u8d77[,\uff0c]?\\s*\u6211",
    "\u5f88\u62b1\u6b49[,\uff0c]?\\s*\u6211",
    "\u62b1\u6b49[,\uff0c]?\\s*\u6211",
    "^\u6211\u65e0\u6cd5",
    "^\u6211\u4e0d\u80fd",
    "^\u65e0\u6cd5",
    "^\u6211\u6ca1\u6709",
    "\u4f5c\u4e3a(\u4e00\u4e2a)?\u4eba\u5de5\u667a\u80fd",
    "\u4f5c\u4e3a(\u4e00\u4e2a)?AI",
    "\u6211\u4e0d\u4f1a(\u8bc4\u4ef7|\u8bc4\u8bba|\u63d0\u4f9b)",
    "\u6211\u4e0d\u4fbf(\u8bc4\u4ef7|\u8bc4\u8bba|\u8ba8\u8bba)",
    "\u4e0d\u4e88\u7f6e\u8bc4",
    "\u4e0d\u4fbf\u8bc4\u4ef7",
    "\u4e0d\u8bc4\u4ef7"
)

EN_PATTERNS <- c(
    "^\\s*(?:i'?m|i am)\\s+(?:sorry|afraid|unable|not able)",
    "^\\s*sorry[,]?\\s+(?:i|but)",
    "^\\s*i\\s+(?:can'?t|cannot|won'?t|am unable|am not able|don'?t)\\s+(?:answer|provide|comment|evaluate|discuss|help|engage|share|give|make)",
    "^\\s*i\\s+(?:must|need to|have to)\\s+(?:decline|refuse)",
    "^\\s*i\\s+(?:will not|won'?t)\\s+(?:answer|provide|comment|evaluate|discuss)",
    "^\\s*(?:as|being)\\s+an?\\s+(?:ai|language model|llm|assistant)[,.\\s]",
    "^\\s*unfortunately[,]?\\s+i\\s+(?:can'?t|cannot|am unable)",
    "^\\s*i\\s+(?:do not|don'?t)\\s+have\\s+(?:an opinion|opinions|personal)",
    "^\\s*i\\s+can'?t\\s+help\\s+with\\s+that",
    "^\\s*it'?s\\s+not\\s+appropriate",
    "^\\s*it\\s+is\\s+not\\s+appropriate",
    "^\\s*(?:that|this)\\s+is\\s+(?:not|too)\\s+(?:appropriate|sensitive)"
)

CN_RE <- paste(CN_PATTERNS, collapse = "|")
EN_RE <- paste(EN_PATTERNS, collapse = "|")

HEAD_CHARS <- 120

is_refusal_cn <- function(text) {
    if (!is.character(text) || length(text) == 0 || is.na(text[1])) return(FALSE)
    if (nchar(trimws(text[1])) == 0) return(FALSE)
    head_text <- substr(text[1], 1, HEAD_CHARS)
    grepl(CN_RE, head_text, perl = TRUE)
}

is_refusal_en <- function(text) {
    if (!is.character(text) || length(text) == 0 || is.na(text[1])) return(FALSE)
    if (nchar(trimws(text[1])) == 0) return(FALSE)
    head_text <- substr(text[1], 1, HEAD_CHARS)
    grepl(EN_RE, head_text, perl = TRUE, ignore.case = TRUE)
}

is_refusal_any <- function(text) {
    is_refusal_cn(text) || is_refusal_en(text)
}

# Vectorized counterparts for use in dplyr pipelines
is_refusal_cn_v <- function(texts) vapply(texts, is_refusal_cn, logical(1))
is_refusal_en_v <- function(texts) vapply(texts, is_refusal_en, logical(1))
is_refusal_any_v <- function(texts) vapply(texts, is_refusal_any, logical(1))
