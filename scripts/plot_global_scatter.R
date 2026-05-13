#!/usr/bin/env Rscript
# Render the cross-national WPFI scatter as a PDF, matching the OJS plot on
# global.qmd. China is shown as a square baseline, the 37 language-exclusive
# target countries as circles. Pooling uses the same Wilson-CI-on-summed-counts
# approach as the OJS code.
#
# Usage:
#   Rscript scripts/plot_global_scatter.R              # all models pooled
#   Rscript scripts/plot_global_scatter.R "GPT-5.5"    # single model

library(tidyverse)
library(jsonlite)
library(ggrepel)

args <- commandArgs(trailingOnly = TRUE)
model_filter <- if (length(args) >= 1) args[[1]] else NULL

.script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
  }
)
proj_dir <- dirname(.script_dir)
in_path <- file.path(proj_dir, "data", "global", "country_scores.json")
out_dir <- file.path(proj_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
scores <- fromJSON(in_path)

if (!is.null(model_filter)) {
  if (!model_filter %in% scores$model) {
    stop(sprintf("Model '%s' not found. Available: %s",
                 model_filter, paste(sort(unique(scores$model)), collapse = ", ")))
  }
  scores <- scores %>% filter(model == model_filter)
  slug <- tolower(gsub("[^A-Za-z0-9]+", "-", model_filter))
  out_path <- file.path(out_dir, paste0("global_scatter_", slug, ".pdf"))
  cat(sprintf("Filtered to model '%s' (%d rows)\n", model_filter, nrow(scores)))
} else {
  out_path <- file.path(out_dir, "global_scatter_all_models.pdf")
}

# Pool across all models: sum favorable + total counts per country, then Wilson CI.
wilson_ci <- function(fav, n, z = 1.96) {
  p <- fav / n
  denom <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denom
  margin <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom
  list(prop = p, lo = pmax(0, center - margin), hi = pmin(1, center + margin))
}

pooled <- scores %>%
  mutate(favorable = round(prop_favorable * n)) %>%
  group_by(country, wpfi_score, situation, target_lang) %>%
  summarise(favorable = sum(favorable), n = sum(n), .groups = "drop") %>%
  mutate(
    prop_favorable = favorable / n,
    ci_lo = wilson_ci(favorable, n)$lo,
    ci_hi = wilson_ci(favorable, n)$hi,
    is_china = country == "China"
  )

cat_colors <- c(
  "Good" = "#4daf4a",
  "Satisfactory" = "#a6d854",
  "Problematic" = "#ffcc00",
  "Difficult" = "#ff7f00",
  "Very Serious" = "#e41a1c"
)
cat_order <- c("Very Serious", "Difficult", "Problematic", "Satisfactory", "Good")

bands <- tibble(
  situation = factor(cat_order, levels = cat_order),
  x0 = c(20, 40, 55, 70, 85),
  x1 = c(40, 55, 70, 85, 95)
)

pooled$situation <- factor(pooled$situation, levels = cat_order)

p <- ggplot() +
  geom_rect(
    data = bands,
    aes(xmin = x0, xmax = x1, ymin = 0.2, ymax = 1.0, fill = situation),
    alpha = 0.10, show.legend = FALSE
  ) +
  geom_hline(yintercept = 0.5, color = "#999", linetype = "dashed") +
  geom_point(
    data = pooled,
    aes(x = wpfi_score, y = prop_favorable, color = situation),
    shape = 16, size = 2.6
  ) +
  geom_text_repel(
    data = pooled,
    aes(x = wpfi_score, y = prop_favorable, label = country),
    size = 2.6, color = "#333",
    max.overlaps = Inf, box.padding = 0.25, point.padding = 0.15,
    segment.color = "#bbb", segment.size = 0.2, min.segment.length = 0.2
  ) +
  scale_color_manual(values = cat_colors, breaks = cat_order, name = NULL) +
  scale_fill_manual(values = cat_colors, breaks = cat_order) +
  guides(color = guide_legend(override.aes = list(shape = 16, size = 3))) +
  scale_x_continuous(limits = c(20, 95), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.2, 1.0), expand = c(0, 0),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(
    x = "World Press Freedom Index Score",
    y = "Prop. favorable in target language",
    caption = if (is.null(model_filter)) {
      "China is plotted as a Study 4 baseline; 37 other points are language-exclusive Study 6 countries. All models pooled."
    } else {
      sprintf("China is plotted as a Study 4 baseline; 37 other points are language-exclusive Study 6 countries. Model: %s.", model_filter)
    }
  ) +
  theme_minimal(base_family = "Helvetica") +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.caption = element_text(hjust = 0, color = "#555", size = 8)
  )

quartz(file = out_path, type = "pdf", width = 9, height = 5.5)
print(p)
dev.off()
cat("Wrote", out_path, "\n")
cat("Countries plotted:", nrow(pooled), "(China included:", any(pooled$is_china), ")\n")
