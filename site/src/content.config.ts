/**
 * Content Collection: proyectos
 *
 * Stores all portfolio projects. Two seccion buckets feed different surfaces:
 *   - top12: rendered on home grouped by eje (4 ejes × variable count = 12 total)
 *   - archive: rendered on /archive as compact cards (currently 11)
 *
 * Eje semantics (mutually exclusive):
 *   - investigacion : papers, modelos, hardware experimental, FFT, GNN, SHM
 *   - ia-campo      : visión por computadora desplegada en obra/campo, edge AI, sensores
 *   - plataformas   : SaaS internos / web apps que usan equipos reales (AECODE, Gen+)
 *   - autoria       : trabajo de autoría multidisciplinar (proyectos personales / juegos / suites)
 *
 * INVARIANT: Markdown filenames (slugs) must be globally unique across BOTH
 * `proyectos/` and `proyectos/archive/`, because the URL strategy is
 * `/proyectos/<leaf-basename>` (subdir is collapsed by ProjectCard.astro).
 * Two files with the same basename in different subdirs would collide on
 * getStaticPaths in `[...slug].astro` and break the build.
 */
import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const proyectos = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/proyectos" }),
  schema: ({ image }) => z.object({
    titulo: z.string(),
    eje: z.enum(["investigacion", "ia-campo", "plataformas", "autoria"]),
    seccion: z.enum(["top12", "archive"]),
    anio: z.number().int().min(2020).max(2030),
    stack: z.array(z.string()).default([]),
    metricas: z.array(z.object({ valor: z.string(), label: z.string() })).default([]),
    enlaces: z.array(z.object({ label: z.string(), url: z.string().url() })).default([]),
    hero: image().optional(),
    galeria: z.array(image()).default([]),
    excerpt: z.string().min(20).max(280),
    orden: z.number().int().default(99),
    publico: z.boolean().default(true),
  }),
});

export const collections = { proyectos };
