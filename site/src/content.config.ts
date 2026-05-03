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
