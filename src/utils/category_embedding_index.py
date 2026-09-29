"""TF-IDF similarity index over the 18 math categories' skill.md summaries.

Default (degraded) implementation: TF-IDF char-ngram cosine similarity (local,
no API). If an embedding API is available later, replace encode()/_build() —
the interface stays the same.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategoryEmbeddingIndex:
    def __init__(self, skills_loader):
        self.skills_loader = skills_loader
        self._vectorizer = None
        self._category_vectors = None  # scipy sparse (n_cats, n_features)
        self._categories = None
        self._build()

    @staticmethod
    def _extract_summary(skill_doc: str, max_chars: int = 1500) -> str:
        """Title + headings + short lines (core concepts), capped at max_chars."""
        lines = skill_doc.splitlines()
        summary = []
        chars = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if (s.startswith("#") or s.startswith(">")
                    or s.startswith("- **")
                    or (1 < len(s) <= 60 and not s.startswith("|") and not s.startswith("-"))):
                summary.append(s)
                chars += len(s)
                if chars >= max_chars:
                    break
        return "\n".join(summary) if summary else skill_doc[:max_chars]

    def _build(self):
        cats = self.skills_loader.categories
        self._categories = cats
        summaries = [self._extract_summary(self.skills_loader.get_skill_document(c)) for c in cats]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._category_vectors = self._vectorizer.fit_transform(summaries)

    def encode(self, text: str):
        return self._vectorizer.transform([text])

    def get_category_vector(self, category: str):
        idx = self._categories.index(category)
        return self._category_vectors[idx]

    def rank_all(self, problem: str, top_k: int = 5):
        """Top-k categories by TF-IDF cosine similarity to the problem (over all 18)."""
        qv = self.encode(problem)
        sims = cosine_similarity(qv, self._category_vectors).flatten()
        ranked = sorted(zip(self._categories, [float(s) for s in sims]), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank(self, problem: str, candidates, top_k: int = 3):
        """Re-rank a candidate list [(name, score), ...] by TF-IDF similarity."""
        qv = self.encode(problem)
        scored = []
        for name, _ in candidates:
            idx = self._categories.index(name)
            sim = float(cosine_similarity(qv, self._category_vectors[idx]).flatten()[0])
            scored.append((name, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

