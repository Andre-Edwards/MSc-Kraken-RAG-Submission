export type IndexStrategy = 'fixed_size' | 'structure_aware';
export type Strategy = IndexStrategy | 'hybrid';

export interface User {
  id: number;
  email: string;
  full_name?: string;
  role: 'admin' | 'user';
}

export interface RetrievalSettings {
  chunking_strategy: Strategy;
  top_k: number;
  run_judge: boolean;
  metadata_rerank_enabled: boolean;
  human_audit_enabled: boolean;
  human_audit_interval: number;
  human_audit_low_score_threshold: number;
  human_audit_max_per_session: number;
  human_audit_cooldown_minutes: number;
}

export interface Citation {
  label: string;
  chunk_id: string;
  file_name: string;
  source_path?: string | null;
  document: string;
  section_title: string;
  page_start: number;
  page_end: number;
  score: number;
  vector_score?: number | null;
  metadata_boost?: number | null;
  source_strategy?: IndexStrategy | null;
}

export interface Evidence extends Citation {
  text: string;
}

export interface ChatResponse {
  chat_log_id: number;
  session_id: number;
  question: string;
  answer: string;
  chunking_strategy: Strategy;
  citations: Citation[];
  evidence: Evidence[];
  judge?: Record<string, unknown> | null;
  retrieval_score: number;
  refused: boolean;
  top_k?: number;
  run_judge?: boolean;
  metadata_rerank_enabled?: boolean;
  retrieved_count?: number;
  corpus_version?: number | null;
  index_version?: number | null;
  human_audit_prompt?: {
    show: boolean;
    reason: string;
    reviewed?: boolean;
    query_count?: number;
    interval?: number;
    low_score_threshold?: number;
    max_per_session?: number;
    cooldown_minutes?: number;
    llm_overall_score?: number | null;
  };
  latency_ms?: {
    embedding?: number | null;
    vector_query?: number | null;
    generation?: number | null;
    judge?: number | null;
    total?: number | null;
  };
  models?: {
    embedding?: string | null;
    generation?: string | null;
    judge?: string | null;
  };
}

export interface ChatHistoryItem {
  id: number;
  session_id: number;
  question: string;
  answer: string;
  chunking_strategy: Strategy;
  citations: Citation[];
  evidence?: Evidence[];
  judge?: Record<string, unknown> | null;
  retrieval_score: number;
  refused: boolean;
  top_k?: number;
  run_judge?: boolean;
  retrieved_count?: number;
  embedding_model?: string | null;
  generation_model?: string | null;
  judge_model?: string | null;
  embedding_latency_ms?: number | null;
  vector_query_latency_ms?: number | null;
  generation_latency_ms?: number | null;
  judge_latency_ms?: number | null;
  total_latency_ms?: number | null;
  corpus_version?: number | null;
  index_version?: number | null;
  human_audit_prompted?: number | null;
  human_audit_prompt_reason?: string | null;
  created_at: string;
}

export interface ChatHistorySession {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  latest_question?: string | null;
}

export interface StatusResponse {
  pdf_dir: string;
  pdf_dir_exists: boolean;
  web_corpus_dir?: string;
  web_pages_count?: number;
  openai_key_configured: boolean;
  index_counts: Record<IndexStrategy, number>;
  corpus_version?: number;
  index_version?: number;
}

export interface DocumentItem {
  file_name: string;
  bytes: number;
  source_type?: 'pdf' | 'web';
  url?: string;
}

export interface WebCrawlResponse {
  ok: boolean;
  pages_saved: number;
  visited: number;
  skipped: Array<{ url: string; reason: string }>;
  output_path: string;
  allowed_domains: string[];
  seed_urls: string[];
  max_pages: number;
  max_depth: number;
  corpus_version: number;
}

export interface WebDuplicatePage {
  title: string;
  url: string;
  bytes: number;
}

export interface WebDuplicateGroup {
  type: 'exact' | 'near';
  similarity: number;
  pages: WebDuplicatePage[];
}

export interface WebDuplicateReport {
  total_pages: number;
  similarity_threshold: number;
  duplicate_group_count: number;
  duplicate_groups: WebDuplicateGroup[];
}

export interface HumanAuditPayload {
  chat_log_id: number;
  groundedness_score: number;
  citation_score: number;
  relevance_score: number;
  completeness_score: number;
  clarity_score: number;
  overall_verdict: 'pass' | 'partial' | 'fail' | 'unable_to_judge';
  issue_tags: string[];
  comment?: string;
  reviewer_confidence: 'low' | 'medium' | 'high';
}

export interface DemoFeedbackPayload {
  role_department?: string;
  ease_score: number;
  helpfulness_score: number;
  trustworthiness_score: number;
  citations_useful: 'yes' | 'somewhat' | 'no';
  incorrect_or_misleading: boolean;
  incorrect_notes?: string;
  missing_expectation?: string;
  improvement?: string;
  would_use_at_work: 'yes' | 'maybe' | 'no';
  final_comments?: string;
}

export interface AdminMetrics {
  total_chats: number;
  active_users: number;
  avg_retrieval_score?: number | null;
  refusals: number;
  human_audit_prompts: number;
  total_human_audits: number;
  avg_human_score?: number | null;
  avg_llm_score_snapshot?: number | null;
  human_verdicts: {
    pass: number;
    partial: number;
    fail: number;
    unable_to_judge: number;
  };
  citation_clicks: number;
}

export type InsightIssueType = 'refused' | 'low_llm' | 'low_human' | 'missing_document';

export interface InsightResolutionFields {
  issue_type: InsightIssueType;
  item_key: string;
  resolved: boolean;
  resolved_at?: string | null;
  resolved_by_email?: string | null;
  resolution_note?: string | null;
}

export interface InsightQuestionItem extends InsightResolutionFields {
  chat_log_id: number;
  question: string;
  answer_preview: string;
  category: string;
  strategy: Strategy;
  created_at: string;
  user_email: string;
  llm_score?: number | null;
  llm_verdict?: string | null;
  llm_notes?: string | null;
}

export interface LowHumanAuditInsight extends InsightResolutionFields {
  audit_id: number;
  chat_log_id: number;
  question: string;
  answer_preview: string;
  category: string;
  strategy: Strategy;
  created_at: string;
  user_email: string;
  human_average_score: number;
  overall_verdict: string;
  reviewer_confidence: string;
  comment?: string | null;
  issue_tags: string[];
  llm_overall_score?: number | null;
}

export interface AdminInsights {
  generated_at: string;
  include_resolved?: boolean;
  resolved_counts?: Record<InsightIssueType, number>;
  summary: {
    total_questions_analyzed: number;
    total_refusals: number;
    low_llm_score_questions: number;
    low_human_score_questions: number;
    corpus_gap_topics: number;
  };
  top_question_categories: Array<{
    category: string;
    total: number;
    refusals: number;
    low_llm: number;
    low_human: number;
    avg_llm_score?: number | null;
    avg_human_score?: number | null;
  }>;
  unanswered_or_refused_questions: InsightQuestionItem[];
  low_llm_judge_score_questions: InsightQuestionItem[];
  low_human_audit_score_questions: LowHumanAuditInsight[];
  common_missing_documents: Array<InsightResolutionFields & {
    topic: string;
    count: number;
    category_total?: number;
    warning_rate?: number | null;
    representative_question?: string | null;
    signals: string[];
    action_type?: string;
    suggested_action?: string;
  }>;
  llm_generated_insights?: {
    available: boolean;
    headline: string;
    executive_summary: string;
    key_takeaways: string[];
    suggested_actions: string[];
    watchlist: string[];
    model?: string | null;
    generated_at?: string | null;
    usage?: {
      prompt_tokens?: number | null;
      completion_tokens?: number | null;
      total_tokens?: number | null;
    };
    error?: string | null;
  };
  rag_evaluation_note: string;
}

export interface CitationClickPayload {
  chat_log_id?: number | null;
  citation_label?: string | null;
  chunk_id?: string | null;
  file_name: string;
  page_start?: number | null;
  page_end?: number | null;
  source_strategy?: string | null;
}
