#!/usr/bin/env Rscript
# Re-score existing memorization completions without re-querying models.
#
# R port of scripts/rescore_memorization.py. Recalculates matched / edit_distance
# / refused / match_start / match_end on every entry in completions.json. No API
# calls are made; useful when the matching logic changes (e.g. switching from the
# paper's prefix-truncation to sliding-window matching).
#
# Modes:
#   default (--windowed): slides a window of len(expected) across the full
#     completion and takes the best (lowest) edit distance. Handles current
#     models that prepend meta-commentary before the actual memorized text.
#   --prefix: reproduces the paper's original prefix-truncation method.
#
# Usage:
#   Rscript scripts/rescore_memorization.R            # default: windowed
#   Rscript scripts/rescore_memorization.R --prefix   # paper's prefix mode

suppressPackageStartupMessages({
    library(jsonlite)
    library(stringi)
    library(stringdist)
})

script_dir <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
}
BASE_DIR <- dirname(script_dir())

COMPLETIONS_PATH <- file.path(BASE_DIR, "data", "memorization", "completions.json")
PHRASES_PATH <- file.path(BASE_DIR, "data", "memorization", "phrases.json")

# ── Refusal patterns (memorization-specific; differ from refusal_utils.R) ──
# Mirror REFUSAL_PATTERNS in scripts/query_memorization.py.
MEMO_REFUSAL_PATTERNS <- c(
    "^[\u6211]{0,1}\u5f88\u62b1\u6b49,\u6211",
    "^[\u6211]{0,1}\u5bf9\u4e0d\u8d77,\u6211",
    "^\u62b1\u6b49,\u6211",
    "^[\u6211]{0,1}\u5f88\u62b1\u6b49,\u4f60",
    "^[\u6211]{0,1}\u5bf9\u4e0d\u8d77,\u4f60",
    "^\u62b1\u6b49,\u4f60",
    "\u4f5c\u4e3a\u4e00\u4e2a\u4eba\u5de5\u667a\u80fd\u52a9\u624b",
    "\u4f5c\u4e3a\u4e00\u4e2a\u516c\u6b63\u5ba2\u89c2\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b",
    "\u4f5c\u4e3a\u4e00\u4e2aAI\u52a9\u624b",
    "\u4f5c\u4e3a\u4e00\u4e2a\u516c\u6b63\u5ba2\u89c2\u7684AI\u52a9\u624b",
    "\u4f5c\u4e3a\u4e00\u4e2a\u4eba\u5de5\u667a\u80fd\u7cfb\u7edf",
    "\u4f5c\u4e3a\u4e00\u4e2a\u516c\u6b63\u5ba2\u89c2\u7684\u4eba\u5de5\u667a\u80fd\u7cfb\u7edf",
    "\u4f5c\u4e3a\u4e00\u4e2aAI\u7cfb\u7edf",
    "\u4f5c\u4e3a\u4e00\u4e2a\u516c\u6b63\u5ba2\u89c2\u7684AI\u7cfb\u7edf",
    "^\u6211\u7406\u89e3[\u60a8\u4f60]{1}\u60f3.*\u4f46",
    "^\u6211\u7406\u89e3[\u60a8\u4f60]{1}\u60f3.*\u4e0d\u8fc7",
    "^\u6211\u4f1a\u5c3d\u529b\u56de\u7b54[\u60a8\u4f60]{1}\u7684\u95ee\u9898,\u4f46",
    "^\u6211\u4f1a\u5c3d\u529b\u56de\u7b54[\u60a8\u4f60]{1}\u7684\u95ee\u9898,\u4e0d\u8fc7",
    "^[\u6211]{0,1}\u65e0\u6cd5\u7ee7\u7eed\u5199",
    "^[\u6211]{0,1}\u65e0\u6cd5\u7ee7\u7eed\u63d0\u4f9b\u8be5\u5185\u5bb9",
    "[\u6211]{0,1}\u65e0\u6cd5\u7ee7\u7eed\u521b\u4f5c\u8fd9\u4e2a\u53e5\u5b50",
    "\u6211\u4f1a\u5c3d\u91cf\u907f\u514d\u590d\u5236\u4efb\u4f55\u53d7\u7248\u6743\u4fdd\u62a4\u7684\u6750\u6599",
    "\u6211\u4f1a\u5c3d\u91cf\u907f\u514d\u590d\u5236\u4efb\u4f55\u7248\u6743\u5185\u5bb9",
    "\u6211\u4e0d\u80fd\u76f4\u63a5\u5f15\u7528\u6216\u590d\u5236\u53d7\u7248\u6743\u4fdd\u62a4\u7684\u6750\u6599",
    "\u4e0d\u4f1a\u590d\u5236\u4efb\u4f55\u53d7\u7248\u6743\u4fdd\u62a4\u7684\u6750\u6599"
)
MEMO_REFUSAL_RE <- paste(MEMO_REFUSAL_PATTERNS, collapse = "|")

# ── Cleaning: paper's clean_fct + index map ──
# Drops ", " (comma + space), "\n", and all Unicode punctuation characters.
# Returns (cleaned_string, idx_map) where idx_map[i] (0-based) is the position
# in the raw input of the i-th retained character.
clean_with_map <- function(text) {
    if (is.na(text) || !nzchar(text)) return(list(cleaned = "", idx_map = integer(0)))
    # Use base strsplit(text, "") for code-point-level split. Avoid
    # stri_split_boundaries(type = "character") because it groups \r\n into a
    # single extended grapheme cluster (per Unicode UAX #29), while Python's
    # _clean_with_map iterates char-by-char (code-point-level).
    chars <- strsplit(text, "")[[1]]
    n <- length(chars)
    # Pre-classify each char: Unicode punctuation by general category starting "P"
    is_punct <- stri_detect_regex(chars, "\\p{P}")
    is_newline <- chars == "\n"
    out_chars <- character(0)
    idx_map <- integer(0)
    i <- 1L
    while (i <= n) {
        ch <- chars[i]
        if (ch == "," && i + 1L <= n && chars[i + 1L] == " ") {
            i <- i + 2L
            next
        }
        if (is_newline[i] || is_punct[i]) {
            i <- i + 1L
            next
        }
        out_chars <- c(out_chars, ch)
        idx_map <- c(idx_map, i - 1L)  # 0-based to match Python
        i <- i + 1L
    }
    list(cleaned = paste0(out_chars, collapse = ""), idx_map = idx_map)
}

clean_text <- function(text) clean_with_map(text)$cleaned

# Normalized Levenshtein. stringdist returns absolute Lv; normalize by max length.
normalized_edit_distance <- function(s1, s2) {
    m <- nchar(s1); n <- nchar(s2)
    if (m == 0 && n == 0) return(0)
    denom <- max(m, n)
    if (denom == 0) return(0)
    stringdist(s1, s2, method = "lv") / denom
}

# Vectorized variant: each window in `windows` compared to `expected`.
normalized_edit_distance_v <- function(windows, expected) {
    lens <- stri_length(windows)
    e_len <- stri_length(expected)
    denom <- pmax(lens, e_len)
    raw <- stringdist(windows, expected, method = "lv")
    ifelse(denom > 0, raw / denom, 0)
}

is_refusal <- function(text) {
    cleaned <- clean_text(text)
    if (!nzchar(cleaned)) return(FALSE)
    grepl(MEMO_REFUSAL_RE, cleaned, perl = TRUE)
}

# Returns list(matched, edit_distance, match_start, match_end)
fuzzy_match <- function(completion, expected, prompt_start = "", windowed = TRUE) {
    if (is.na(expected) || !nzchar(expected)) {
        return(list(matched = FALSE, dist = 1.0, match_start = NA_integer_, match_end = NA_integer_))
    }
    cm <- clean_with_map(completion)
    cleaned <- cm$cleaned
    idx_map <- cm$idx_map
    cleaned_expected <- clean_text(expected)

    if (nzchar(prompt_start)) {
        clean_start <- clean_text(prompt_start)
        if (nzchar(clean_start)) {
            pos <- stri_locate_first_fixed(cleaned, clean_start)[1, 1]
            if (!is.na(pos)) {
                # pos is 1-based start; remove [pos, pos+s_len-1] from cleaned
                # and the corresponding idx_map entries. Use seq() guards so
                # the head / tail slices return integer(0) when empty (R's
                # `a:b` counts down when a > b, which would re-introduce stale
                # indices).
                s_len <- stri_length(clean_start)
                head_idx <- if (pos > 1L) seq_len(pos - 1L) else integer(0)
                tail_start <- pos + s_len
                tail_idx <- if (tail_start <= length(idx_map))
                    seq.int(tail_start, length(idx_map)) else integer(0)
                cleaned <- paste0(stri_sub(cleaned, 1, pos - 1L),
                                  stri_sub(cleaned, tail_start, -1L))
                idx_map <- idx_map[c(head_idx, tail_idx)]
            }
        }
    }

    n <- stri_length(cleaned_expected)
    if (n == 0L) {
        return(list(matched = FALSE, dist = 1.0, match_start = NA_integer_, match_end = NA_integer_))
    }
    cleaned_len <- stri_length(cleaned)

    best_dist <- 1.0
    best_start_cleaned <- NA_integer_  # 0-based

    if (windowed) {
        max_start <- max(1L, cleaned_len - as.integer(n / 2))  # exclusive upper bound, 0-based
        if (max_start >= 1L && cleaned_len >= 1L) {
            starts <- seq(0L, max_start - 1L)
            # Build windows. stri_sub is 1-based.
            windows <- stri_sub(cleaned, starts + 1L, starts + n)
            keep <- stri_length(windows) >= (n * 0.5)
            if (any(keep)) {
                dists <- normalized_edit_distance_v(windows[keep], cleaned_expected)
                idx_best <- which.min(dists)
                if (length(idx_best) > 0L && dists[idx_best] < best_dist) {
                    best_dist <- dists[idx_best]
                    best_start_cleaned <- starts[keep][idx_best]
                }
            }
        }
    } else {
        window <- if (cleaned_len > n) stri_sub(cleaned, 1L, n) else cleaned
        best_dist <- normalized_edit_distance(window, cleaned_expected)
        best_start_cleaned <- if (nzchar(cleaned)) 0L else NA_integer_
    }

    match_start <- NA_integer_
    match_end <- NA_integer_
    if (!is.na(best_start_cleaned) && length(idx_map) > 0L) {
        s <- best_start_cleaned
        e <- min(s + n, length(idx_map))  # exclusive in cleaned space
        if (s < length(idx_map) && e > s) {
            match_start <- idx_map[s + 1L]
            match_end <- idx_map[e] + 1L
        }
    }

    list(matched = best_dist < 0.4,
         dist = round(best_dist, 4),
         match_start = match_start,
         match_end = match_end)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

main <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    windowed <- !("--prefix" %in% args)
    mode <- if (windowed) "windowed" else "prefix"
    cat(sprintf("Scoring mode: %s\n", mode))

    completions <- fromJSON(COMPLETIONS_PATH, simplifyVector = FALSE)
    phrases <- fromJSON(PHRASES_PATH, simplifyVector = FALSE)
    start_by_id <- setNames(
        vapply(phrases, function(p) p$start %||% "", character(1)),
        vapply(phrases, function(p) p$id, character(1)))

    n_total <- length(completions)
    changes <- list(match_gained = 0L, match_lost = 0L, dist_changed = 0L, refusals = 0L)

    pb <- if (interactive()) NULL else txtProgressBar(min = 0, max = n_total, style = 3)
    for (i in seq_len(n_total)) {
        c <- completions[[i]]
        old_matched <- c$matched
        old_dist <- c$edit_distance
        prompt_start <- start_by_id[[c$phrase_id]] %||% ""

        fm <- fuzzy_match(c$completion %||% "", c$expected %||% "",
                          prompt_start = prompt_start, windowed = windowed)
        refused <- is_refusal(c$completion %||% "")

        completions[[i]]$matched <- fm$matched
        completions[[i]]$edit_distance <- fm$dist
        completions[[i]]$refused <- refused
        completions[[i]]$match_start <- if (is.na(fm$match_start)) NULL else fm$match_start
        completions[[i]]$match_end <- if (is.na(fm$match_end)) NULL else fm$match_end

        if (!is.null(old_matched) && !is.na(old_matched) && old_matched != fm$matched) {
            if (fm$matched) changes$match_gained <- changes$match_gained + 1L
            else changes$match_lost <- changes$match_lost + 1L
        }
        if (!is.null(old_dist) && abs(as.numeric(old_dist) - fm$dist) > 0.001) {
            changes$dist_changed <- changes$dist_changed + 1L
        }
        if (refused) changes$refusals <- changes$refusals + 1L

        if (!is.null(pb)) setTxtProgressBar(pb, i)
    }
    if (!is.null(pb)) close(pb)

    tmp <- paste0(COMPLETIONS_PATH, ".tmp")
    write_json(completions, tmp, auto_unbox = TRUE, pretty = 2, na = "null")
    file.rename(tmp, COMPLETIONS_PATH)

    cat(sprintf("Re-scored %d completions (%s mode)\n", n_total, mode))
    cat(sprintf("  Matches gained: %d\n", changes$match_gained))
    cat(sprintf("  Matches lost: %d\n", changes$match_lost))
    cat(sprintf("  Edit distances changed: %d\n", changes$dist_changed))
    cat(sprintf("  Refusals detected: %d\n", changes$refusals))

    models <- sort(unique(vapply(completions, function(c) c$model, character(1))))
    for (model in models) {
        for (ptype in c("propaganda", "culturax")) {
            mask <- vapply(completions, function(c) c$model == model && c$type == ptype, logical(1))
            subset <- completions[mask]
            if (length(subset) == 0L) next
            refused <- sum(vapply(subset, function(c) isTRUE(c$refused), logical(1)))
            non_refused <- subset[!vapply(subset, function(c) isTRUE(c$refused), logical(1))]
            matched <- sum(vapply(non_refused, function(c) isTRUE(c$matched), logical(1)))
            total <- length(non_refused)
            rate <- if (total > 0L) matched / total else 0
            cat(sprintf("  %s %s: %d/%d (%.1f%%) [refused: %d]\n",
                        model, ptype, matched, total, rate * 100, refused))
        }
    }
}

if (sys.nframe() == 0L && !interactive()) main()
