#!/usr/bin/env Rscript
# Reproduce the "Cross-National Audit" scatterplot from global.qmd
# Input: data/global/country_scores.json
# Output: plots/global_scatterplot.pdf

library(tidyverse)
library(jsonlite)
library(scales)
library(ggrepel)
set.seed(42)

base_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) dirname(dirname(normalizePath(f, mustWork = FALSE))) else getwd()
}, error = function(e) getwd())

scores <- fromJSON(file.path(base_dir, "data/global/country_scores.json")) %>%
  as_tibble()

# Default view: new models, averaged across models per country
new_scores <- scores %>% filter(era == "new")

avg <- new_scores %>%
  group_by(country) %>%
  summarize(
    prop_favorable = mean(prop_favorable),
    wpfi_score = first(wpfi_score),
    situation = first(situation),
    n = sum(n),
    .groups = "drop"
  ) %>%
  # Pooled Wilson CI
  mutate(
    total_fav = round(prop_favorable * n),
    p = total_fav / n,
    z = 1.96,
    denom_w = 1 + z^2 / n,
    center = (p + z^2 / (2 * n)) / denom_w,
    margin = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom_w,
    ci_lo = pmax(0, center - margin),
    ci_hi = pmin(1, center + margin)
  ) %>%
  select(-total_fav, -p, -z, -denom_w, -center, -margin)

cat("Global scores (new models, averaged):\n")
avg %>%
  arrange(wpfi_score) %>%
  mutate(fav_pct = sprintf("%.1f%%", prop_favorable * 100)) %>%
  select(country, wpfi_score, situation, fav_pct, n) %>%
  print(n = 40)

# WPFI category colors and band boundaries
cat_colors <- c(
  "Very Serious" = "#e41a1c",
  "Difficult" = "#ff7f00",
  "Problematic" = "#ffcc00",
  "Satisfactory" = "#a6d854",
  "Good" = "#4daf4a"
)

bands <- tribble(
  ~cat, ~x0, ~x1,
  "Very Serious", 0, 40,
  "Difficult", 40, 55,
  "Problematic", 55, 70,
  "Satisfactory", 70, 85,
  "Good", 85, 100
)

avg <- avg %>%
  mutate(situation = factor(situation, levels = names(cat_colors)))

p <- ggplot(avg, aes(x = wpfi_score, y = prop_favorable)) +
  # Background bands
  geom_rect(data = bands, inherit.aes = FALSE,
            aes(xmin = x0, xmax = x1, ymin = 0.2, ymax = 1.0, fill = cat),
            alpha = 0.08) +
  scale_fill_manual(values = cat_colors, guide = "none") +
  # Reference line
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#999") +
  # Points
  geom_point(aes(color = situation), size = 2.5, stroke = 0.3, shape = 21,
             fill = NA) +
  geom_point(aes(color = situation), size = 2) +
  scale_color_manual(values = cat_colors, name = NULL,
                     breaks = names(cat_colors)) +
  # Country labels

  geom_text_repel(aes(label = country), size = 2.5, color = "#333",
                  max.overlaps = 40, segment.color = NA,
                  box.padding = 0.3, point.padding = 0.2) +
  scale_x_continuous(limits = c(20, 95)) +
  scale_y_continuous(limits = c(0.2, 1.0)) +
  labs(x = "World Press Freedom Index Score \u2192",
       y = "\u2191 Prop. favorable in target language") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave(file.path(base_dir, "plots/global_scatterplot.pdf"), p, width = 8, height = 5)
cat("Wrote plots/global_scatterplot.pdf\n")
