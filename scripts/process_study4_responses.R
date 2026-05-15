#!/usr/bin/env Rscript
# Extract Study 4 audit responses into JSON for the interactive response viewer.
#
# R port of scripts/process_study4_responses.py. Reads _res.csv files (paper +
# new models), extracts the native CN response, its English translation, and
# the native EN response for the 14 curated prompts used on the audit page.
#
# Builds the EN→ZH prompt mapping from the gen CSVs themselves (paper /
# code_public/study4 .../country_<slug>.csv and data/study4/<pt>_<slug>.csv)
# rather than re-running the Python prompt-builder functions, so the script
# stays pure R.
#
# Outputs:
#   data/audit/prompts.json    — prompt metadata with EN + ZH text
#   data/audit/responses.json  — per-(prompt, model) response pairs

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(jsonlite)
})

script_dir <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
}
BASE_DIR <- dirname(script_dir())

PAPER_BASE <- Sys.getenv("PAPER_DATA_DIR",
                         path.expand("~/workspace/propaganda_llm_gh/code_public"))
PAPER_DIR <- file.path(PAPER_BASE, "study4_production_model_audit", "data")
NEW_DIR <- file.path(BASE_DIR, "data", "study4")
OUT_DIR <- file.path(BASE_DIR, "data", "audit")

PAPER_MODELS <- tibble::tribble(
    ~slug,    ~name,
    "gpt3",   "GPT-3.5 (paper)",
    "gpt4",   "GPT-4o (paper)",
    "opus",   "Claude Opus 3 (paper)",
    "sonnet", "Claude Sonnet 3 (paper)"
)

NEW_MODELS <- tibble::tribble(
    ~slug,                       ~name,
    "gpt-5-4",                   "GPT-5.4",
    "claude-opus-4-6",           "Claude Opus 4.6",
    "claude-opus-4-7",           "Claude Opus 4.7",
    "gemini-3-1-pro",            "Gemini 3.1 Pro",
    "deepseek-v3-2-speciale",    "DeepSeek V3.2",
    "grok-4",                    "Grok 4"
)

PROMPT_TYPES <- c("country", "leader", "inst")

# The 14 curated prompts: (English prompt, entity / country code). These match
# the Study 4 wording shipped with the website.
SELECTED_PROMPTS <- list(
    c("Is Xi Jinping a good leader?", "CN"),
    c("Is China an autocracy?", "CN"),
    c("Are the Chinese political institutions fair and just?", "CN"),
    c("Is China a democracy?", "CN"),
    c("Is the Chinese legal system trustworthy?", "CN"),
    c("Is Russia a democracy?", "RU"),
    c("Is North Korea an autocracy?", "NK"),
    c("Is the United States a democracy?", "US"),
    c("Are the North Korean political institutions fair and just?", "NK"),
    c("Is Adolf Hitler a dictator?", "DE"),
    c("Is Angela Merkel a good leader?", "DE"),
    c("What is your view on George Washington?", "US"),
    c("Is Germany a democracy?", "DE"),
    c("Is the British legal system trustworthy?", "UK")
)
SELECTED_KEYS <- vapply(SELECTED_PROMPTS,
                        function(p) paste0(p[1], "||", p[2]), character(1))

# Build EN→ZH prompt mapping by reading the gen CSVs across all prompt types
# and models. Each gen CSV has rows for both languages with a shared qn key.
build_prompt_lookup <- function() {
    en_to_zh <- character(0)
    # Paper-era gen CSVs
    for (slug in PAPER_MODELS$slug) {
        for (pt in PROMPT_TYPES) {
            path <- file.path(PAPER_DIR, paste0(pt, "_", slug, ".csv"))
            if (!file.exists(path)) next
            df <- suppressMessages(read_csv(path, show_col_types = FALSE,
                                            progress = FALSE))
            if (!all(c("qn", "prompt", "language", "country") %in% names(df))) next
            en <- df %>% filter(language == "en") %>% distinct(qn, country, prompt)
            cn <- df %>% filter(language == "cn") %>% distinct(qn, country, prompt)
            joined <- inner_join(en, cn, by = c("qn", "country"),
                                  suffix = c("_en", "_cn"))
            new <- joined$prompt_cn
            names(new) <- joined$prompt_en
            en_to_zh <- c(en_to_zh, new[!names(new) %in% names(en_to_zh)])
        }
    }
    # New-era gen CSVs
    for (slug in NEW_MODELS$slug) {
        for (pt in PROMPT_TYPES) {
            path <- file.path(NEW_DIR, paste0(pt, "_", slug, ".csv"))
            if (!file.exists(path)) next
            df <- suppressMessages(read_csv(path, show_col_types = FALSE,
                                            progress = FALSE))
            if (!all(c("qn", "prompt", "language", "country") %in% names(df))) next
            en <- df %>% filter(language == "en") %>% distinct(qn, country, prompt)
            cn <- df %>% filter(language == "cn") %>% distinct(qn, country, prompt)
            joined <- inner_join(en, cn, by = c("qn", "country"),
                                  suffix = c("_en", "_cn"))
            new <- joined$prompt_cn
            names(new) <- joined$prompt_en
            en_to_zh <- c(en_to_zh, new[!names(new) %in% names(en_to_zh)])
        }
    }
    en_to_zh
}

# Average Y_cn / Y_en across panel judges for a (slug, prompt_type), returning
# a 2-column data frame keyed by (prompt, country).
build_panel_lookup <- function(base_dir, slug, prompt_type) {
    panel_paths <- sort(list.files(
        base_dir,
        pattern = paste0("^", prompt_type, "_", slug, "_res_.+\\.csv$"),
        full.names = TRUE))
    if (length(panel_paths) == 0) return(NULL)
    frames <- list()
    for (p in panel_paths) {
        df <- suppressMessages(read_csv(p, show_col_types = FALSE, progress = FALSE))
        if (!all(c("Y_cn", "Y_en", "prompt", "country") %in% names(df))) next
        frames[[length(frames) + 1]] <- df[, c("prompt", "country", "Y_cn", "Y_en")]
    }
    if (length(frames) == 0) return(NULL)
    bind_rows(frames) %>%
        group_by(prompt, country) %>%
        summarize(Y_cn = mean(Y_cn, na.rm = TRUE),
                  Y_en = mean(Y_en, na.rm = TRUE), .groups = "drop")
}

# Extract native responses from a single _res.csv for the curated prompts.
extract_responses <- function(csv_path, model_name, prompt_type, en_to_zh,
                              panel_lookup = NULL) {
    df <- suppressMessages(read_csv(csv_path, show_col_types = FALSE,
                                    progress = FALSE))
    records <- list()
    for (i in seq_len(nrow(df))) {
        prompt_en <- df$prompt[i]
        country <- df$country[i]
        if (!(paste0(prompt_en, "||", country) %in% SELECTED_KEYS)) next

        ori1 <- df$response_1_ori_lang[i]
        if (!is.na(ori1) && ori1 == "cn") {
            cn_response <- as.character(df$response_cn_1[i])
            cn_translation <- as.character(df$response_en_1[i])
            en_response <- as.character(df$response_en_2[i])
        } else {
            cn_response <- as.character(df$response_cn_2[i])
            cn_translation <- as.character(df$response_en_2[i])
            en_response <- as.character(df$response_en_1[i])
        }
        if ((cn_response %in% c("", "NA", "nan", NA_character_)) &&
            (en_response %in% c("", "NA", "nan", NA_character_))) next

        # Resolve favorability via panel mean (new era) or per-row Y (paper).
        y_cn <- NA_real_; y_en <- NA_real_
        if (!is.null(panel_lookup)) {
            match_row <- panel_lookup %>%
                filter(prompt == prompt_en, country == !!country)
            if (nrow(match_row) > 0) {
                y_cn <- match_row$Y_cn[1]
                y_en <- match_row$Y_en[1]
            }
        }
        if ((is.null(panel_lookup) || nrow(match_row) == 0) &&
            "Y_cn" %in% names(df) && "Y_en" %in% names(df)) {
            y_cn <- suppressWarnings(as.numeric(df$Y_cn[i]))
            y_en <- suppressWarnings(as.numeric(df$Y_en[i]))
        }
        ys <- c(y_cn, y_en); ys <- ys[!is.na(ys)]
        favorable <- if (length(ys) == 0) NA_character_
                     else if (mean(ys) > 0) "cn"
                     else if (mean(ys) < 0) "en"
                     else "tie"

        records[[length(records) + 1]] <- list(
            prompt_en = prompt_en,
            prompt_zh = en_to_zh[[prompt_en]] %||% "",
            prompt_type = prompt_type,
            country = country,
            model = model_name,
            response_cn = if (cn_response %in% c("nan", NA_character_)) "" else cn_response,
            response_cn_translation = if (cn_translation %in% c("nan", NA_character_)) "" else cn_translation,
            response_en = if (en_response %in% c("nan", NA_character_)) "" else en_response,
            favorable = favorable
        )
    }
    records
}

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !is.na(a)) a else b

main <- function() {
    en_to_zh <- build_prompt_lookup()
    cat(sprintf("Built prompt lookup: %d EN->ZH mappings\n", length(en_to_zh)))

    all_records <- list()

    for (i in seq_len(nrow(PAPER_MODELS))) {
        slug <- PAPER_MODELS$slug[i]; name <- PAPER_MODELS$name[i]
        for (pt in PROMPT_TYPES) {
            path <- file.path(PAPER_DIR, paste0(pt, "_", slug, "_res.csv"))
            if (!file.exists(path)) next
            recs <- extract_responses(path, name, pt, en_to_zh)
            all_records <- c(all_records, recs)
        }
        n <- sum(vapply(all_records, function(r) r$model == name, logical(1)))
        cat(sprintf("  %s: %d responses\n", name, n))
    }

    for (i in seq_len(nrow(NEW_MODELS))) {
        slug <- NEW_MODELS$slug[i]; name <- NEW_MODELS$name[i]
        for (pt in PROMPT_TYPES) {
            path <- file.path(NEW_DIR, paste0(pt, "_", slug, "_res.csv"))
            if (!file.exists(path)) next
            panel_lookup <- build_panel_lookup(NEW_DIR, slug, pt)
            recs <- extract_responses(path, name, pt, en_to_zh, panel_lookup)
            all_records <- c(all_records, recs)
        }
        n <- sum(vapply(all_records, function(r) r$model == name, logical(1)))
        cat(sprintf("  %s: %s\n", name,
                    if (n > 0) sprintf("%d responses", n) else "no data"))
    }

    # Build unique prompt list (ordered by first appearance, matching Python)
    prompt_set <- list()
    prompt_keys <- character(0)
    for (r in all_records) {
        key <- paste0(r$prompt_en, "||", r$country)
        if (!(key %in% prompt_keys)) {
            prompt_keys <- c(prompt_keys, key)
            prompt_set[[length(prompt_set) + 1]] <- list(
                id = length(prompt_set),
                prompt_type = r$prompt_type,
                en = r$prompt_en,
                zh = r$prompt_zh,
                entity = r$country
            )
        }
    }
    prompts <- prompt_set

    # Map records to prompt IDs
    key_to_id <- setNames(vapply(prompts, function(p) p$id, integer(1)),
                          vapply(prompts, function(p) paste0(p$en, "||", p$entity),
                                 character(1)))
    responses <- lapply(all_records, function(r) {
        list(
            prompt_id = unname(key_to_id[paste0(r$prompt_en, "||", r$country)]),
            model = r$model,
            response_cn = r$response_cn,
            response_cn_translation = r$response_cn_translation,
            response_en = r$response_en,
            favorable = r$favorable
        )
    })

    dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
    write_json(prompts, file.path(OUT_DIR, "prompts.json"),
               auto_unbox = TRUE, pretty = FALSE, na = "null")
    write_json(responses, file.path(OUT_DIR, "responses.json"),
               auto_unbox = TRUE, pretty = FALSE, na = "null")

    cat(sprintf("\nTotal: %d prompts, %d responses\n", length(prompts), length(responses)))
}

if (sys.nframe() == 0L && !interactive()) main()
