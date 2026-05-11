#!/usr/bin/env Rscript
# Unified Bradley-Terry over (model, language) players.
#
# Players: 9 SUTs × 2 languages = 18 per focal country.
# Edges:
#   1. Cross-model same-language (from data/cross_model_audit/*.csv): the
#      9-pair Hamiltonian cycle, in both CN and EN. 18 edges per country.
#   2. Within-model cross-language (from data/study4/*_res_*.csv): for each
#      model, two binomial outcomes per prompt (Y_cn = judgment when both
#      responses displayed in CN; Y_en = judgment when both in EN). 9 edges
#      per country, but each contributes ~ 2x prompts × 2 judges binomial
#      outcomes.
#
# Refusal filter: drop rows where the SUT refused in either language,
# consistent with the audit page methodology.
#
# Centering: sum-to-zero across all 18 players per focal country, with
# SEs propagated through the linear shift.
#
# Output: data/cross_model_audit/bt_scores_with_language.json
#   {model, language, focal_country, judge, bt_logit, ci_lo, ci_hi, n_obs}
# Judges: "gpt-oss-120b", "deepseek-v32", and "all" (pooled).

suppressPackageStartupMessages({
    library(BradleyTerry2)
    library(dplyr)
    library(jsonlite)
    library(readr)
    library(stringr)
})

COUNTRY_DISPLAY <- c(
    CN = "China", RU = "Russia", NK = "North Korea",
    US = "United States", UK = "United Kingdom", DE = "Germany"
)

# Display name -> filename slug used in data/study4/{qt}_{slug}.csv
SUT_SLUGS <- list(
    "gpt-5.4" = "gpt-5-4",
    "gpt-5.5" = "gpt-5-5",
    "claude-opus-4.6" = "claude-opus-4-6",
    "claude-opus-4.7" = "claude-opus-4-7",
    "gemini-3.1-pro" = "gemini-3-1-pro",
    "deepseek-v3.2" = "deepseek-v3-2-speciale",
    "deepseek-v4-pro" = "deepseek-v4-pro",
    "grok-4" = "grok-4",
    "grok-4.3" = "grok-4-3"
)
SLUG_TO_MODEL <- setNames(names(SUT_SLUGS), unlist(SUT_SLUGS))
QN_TYPES <- c("leader", "country", "inst")
JUDGES <- c("gpt-oss-120b", "deepseek-v3.2")

# Refusal regexes mirror scripts/refusal_utils.py
CN_REFUSAL <- "对不起,?\\s*我|很抱歉,?\\s*我|抱歉,?\\s*我|^我无法|^我不能|^无法|^我没有|作为(一个)?人工智能|作为(一个)?AI|我不会(评价|评论|提供)|我不便(评价|评论|讨论)|不予置评|不便评价|不评价"
EN_REFUSAL <- paste(
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
    sep = "|"
)
HEAD <- 120

is_refusal_cn <- function(x) {
    if (is.na(x) || nchar(x) == 0) return(FALSE)
    grepl(CN_REFUSAL, substr(x, 1, HEAD))
}
is_refusal_en <- function(x) {
    if (is.na(x) || nchar(x) == 0) return(FALSE)
    grepl(EN_REFUSAL, substr(x, 1, HEAD), ignore.case = TRUE, perl = TRUE)
}
is_refusal_any <- function(x) is_refusal_cn(x) || is_refusal_en(x)

# --- Build refusal lookups per SUT ---
build_refusal_set <- function(model, slug) {
    refused_prompts <- character(0)
    for (qt in QN_TYPES) {
        path <- file.path("data/study4", paste0(qt, "_", slug, ".csv"))
        if (!file.exists(path)) next
        df <- suppressMessages(read_csv(path, show_col_types = FALSE))
        cn <- df %>% filter(language == "cn")
        en <- df %>% filter(language == "en")
        n <- min(nrow(cn), nrow(en))
        prompts_en <- en$prompt[seq_len(n)]
        cn_refused <- sapply(cn$response_cn[seq_len(n)], is_refusal_any, USE.NAMES = FALSE)
        en_refused <- sapply(en$response_en[seq_len(n)], is_refusal_any, USE.NAMES = FALSE)
        refused_idx <- which(cn_refused | en_refused)
        refused_prompts <- c(refused_prompts, prompts_en[refused_idx])
    }
    refused_prompts
}

cat("Building refusal lookups...\n")
refusal_sets <- list()
for (model in names(SUT_SLUGS)) {
    refusal_sets[[model]] <- build_refusal_set(model, SUT_SLUGS[[model]])
    cat(sprintf("  %s: %d refused prompts\n", model, length(refusal_sets[[model]])))
}

# --- Load cross-model (same-language) outcomes ---
cat("Loading cross-model outcomes...\n")
cross_files <- list.files("data/cross_model_audit",
                          pattern = "_vs_.*\\.csv$", full.names = TRUE)
cross_files <- cross_files[!grepl("bt_scores", cross_files)]
cross_df <- bind_rows(lapply(cross_files, function(f) {
    df <- suppressMessages(read_csv(f, show_col_types = FALSE))
    stem <- tools::file_path_sans_ext(basename(f))
    judge_slug <- tail(strsplit(stem, "_")[[1]], 1)
    # Reverse the dot-stripping
    judge <- ifelse(judge_slug == "gpt-oss-120b", "gpt-oss-120b",
            ifelse(judge_slug == "deepseek-v32", "deepseek-v3.2", judge_slug))
    df$judge <- judge
    df
}))
cat(sprintf("  %d cross-model rows\n", nrow(cross_df)))

# Filter cross-model by refusal (drop if either side refused on this prompt)
cross_df <- cross_df %>%
    rowwise() %>%
    mutate(skip = (prompt %in% refusal_sets[[model_a]]) ||
                  (prompt %in% refusal_sets[[model_b]])) %>%
    ungroup() %>%
    filter(!skip, !is.na(winner), winner %in% c("model_a", "model_b"))
cat(sprintf("  %d after refusal filter\n", nrow(cross_df)))

# Build BT-friendly rows: each row is one binomial outcome between two
# (model, language) players. For cross-model: player = (model, language).
cross_long <- cross_df %>%
    mutate(
        winner_model = ifelse(winner == "model_a", model_a, model_b),
        loser_model = ifelse(winner == "model_a", model_b, model_a),
        winner_player = paste0(winner_model, "::", language),
        loser_player = paste0(loser_model, "::", language)
    ) %>%
    select(focal_country = country, judge, winner_player, loser_player)

# --- Load within-model cross-language outcomes from main audit ---
cat("Loading within-model outcomes from data/study4/*_res_*.csv...\n")
within_rows <- list()
for (model in names(SUT_SLUGS)) {
    slug <- SUT_SLUGS[[model]]
    refused <- refusal_sets[[model]]
    for (qt in QN_TYPES) {
        for (judge in JUDGES) {
            path <- file.path("data/study4",
                              paste0(qt, "_", slug, "_res_", judge, ".csv"))
            if (!file.exists(path)) next
            df <- suppressMessages(read_csv(path, show_col_types = FALSE))
            if (!all(c("Y_cn", "Y_en", "prompt") %in% names(df))) next
            # Refusal filter on EN prompt key
            df <- df %>% filter(!(prompt %in% refused))
            # Y_cn / Y_en: +1 = CN-prompt-origin won, -1 = EN-prompt-origin
            for (col in c("Y_cn", "Y_en")) {
                sub <- df %>% select(country, Y = all_of(col)) %>% filter(!is.na(Y), Y != 0)
                if (nrow(sub) == 0) next
                rows <- sub %>% mutate(
                    focal_country = country,
                    judge = judge,
                    winner_player = ifelse(Y == 1,
                                            paste0(model, "::cn"),
                                            paste0(model, "::en")),
                    loser_player = ifelse(Y == 1,
                                           paste0(model, "::en"),
                                           paste0(model, "::cn"))
                ) %>% select(focal_country, judge, winner_player, loser_player)
                within_rows[[length(within_rows) + 1]] <- rows
            }
        }
    }
}
within_long <- bind_rows(within_rows)
cat(sprintf("  %d within-model rows\n", nrow(within_long)))

# --- Combine and fit per (focal_country, judge) stratum ---
all_long <- bind_rows(cross_long, within_long)
cat(sprintf("Total BT input rows: %d\n", nrow(all_long)))

fit_stratum <- function(df, focal_country, judge_label) {
    if (judge_label == "all") {
        sub <- df %>% filter(focal_country == !!focal_country)
    } else {
        sub <- df %>% filter(focal_country == !!focal_country, judge == judge_label)
    }
    if (nrow(sub) < 10) return(NULL)

    counts <- sub %>% count(winner_player, loser_player, name = "wins")
    tab <- xtabs(wins ~ winner_player + loser_player, data = counts)
    pair_counts <- countsToBinomial(tab)
    all_levels <- sort(unique(c(as.character(pair_counts$player1),
                                 as.character(pair_counts$player2))))
    pair_counts$player1 <- factor(pair_counts$player1, levels = all_levels)
    pair_counts$player2 <- factor(pair_counts$player2, levels = all_levels)

    fit <- tryCatch(
        BTm(cbind(win1, win2), player1, player2, data = pair_counts),
        error = function(e) NULL
    )
    if (is.null(fit)) return(NULL)

    ab <- BTabilities(fit)
    n <- nrow(ab)
    players <- rownames(ab)
    raw <- ab[, "ability"]

    # Sum-to-zero recentering with propagated SE
    sigma <- matrix(0, n, n, dimnames = list(players, players))
    vc <- vcov(fit)
    vc_names <- gsub("^\\.\\.", "", rownames(vc))
    rownames(vc) <- vc_names; colnames(vc) <- vc_names
    common <- intersect(players, vc_names)
    if (length(common) > 0) sigma[common, common] <- vc[common, common]
    M <- diag(n) - matrix(1 / n, n, n)
    cov_c <- M %*% sigma %*% t(M)
    se_centered <- sqrt(pmax(diag(cov_c), 0))
    centered <- raw - mean(raw)

    parts <- str_split_fixed(players, "::", 2)
    out <- data.frame(
        model = parts[, 1],
        language = parts[, 2],
        focal_country = focal_country,
        judge = judge_label,
        bt_logit = centered,
        se = se_centered,
        stringsAsFactors = FALSE
    )
    out$ci_lo <- out$bt_logit - 1.96 * out$se
    out$ci_hi <- out$bt_logit + 1.96 * out$se
    out$n_obs <- nrow(sub)
    out
}

countries <- unique(all_long$focal_country)
judges_list <- c(JUDGES, "all")
results <- list()
for (c in countries) {
    for (j in judges_list) {
        results[[length(results) + 1]] <- fit_stratum(all_long, c, j)
    }
}
results <- bind_rows(results)
results$focal_country_display <- COUNTRY_DISPLAY[results$focal_country]

# Round for JSON
results$bt_logit <- round(results$bt_logit, 4)
results$ci_lo <- round(results$ci_lo, 4)
results$ci_hi <- round(results$ci_hi, 4)
results$se <- round(results$se, 4)
# Normalize judge slug to match cross-model JSON conventions
results$judge <- ifelse(results$judge == "deepseek-v3.2", "deepseek-v32",
                 ifelse(results$judge == "gpt-oss-120b", "gpt-oss-120b",
                 results$judge))

out_path <- "data/cross_model_audit/bt_scores_with_language.json"
write_json(results, out_path, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("Wrote %d estimates to %s\n", nrow(results), out_path))
