import type {
  AdminInsights,
  AdminMetrics,
  ChatHistorySession,
  ChatHistoryItem,
  ChatResponse,
  CitationClickPayload,
  DemoFeedbackPayload,
  DocumentItem,
  HumanAuditPayload,
  IndexStrategy,
  InsightIssueType,
  RetrievalSettings,
  StatusResponse,
  Strategy,
  User,
  WebCrawlResponse,
  WebDuplicateReport,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const TOKEN_KEY = 'kraken_rag_token';

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch {
    const destination = API_BASE || 'the local Vite proxy at http://127.0.0.1:8000';
    throw new Error(
      `Cannot reach the backend through ${destination}. Confirm that FastAPI is running and that /api/health opens successfully.`,
    );
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return body as T;
}

export async function login(email: string, password: string): Promise<{ access_token: string; user: User }> {
  return request('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
}

export async function me(): Promise<User> {
  return request('/api/auth/me');
}

export async function getStatus(): Promise<StatusResponse> {
  return request('/api/documents/status');
}

export async function getDocuments(): Promise<{ documents: DocumentItem[]; web_documents?: DocumentItem[]; pdf_dir: string; web_corpus_dir?: string }> {
  return request('/api/documents');
}

export async function crawlWebCorpus(payload: {
  seed_urls: string[];
  allowed_domains?: string[];
  max_pages: number;
  max_depth: number;
}): Promise<WebCrawlResponse> {
  return request('/api/documents/web-crawl', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function detectWebDuplicates(similarityThreshold = 0.86): Promise<WebDuplicateReport> {
  return request(`/api/documents/web-crawl/duplicates?similarity_threshold=${similarityThreshold}`);
}

export async function uploadDocuments(files: File[]): Promise<{ uploaded: DocumentItem[]; pdf_dir: string }> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  return request('/api/documents/upload', {
    method: 'POST',
    body: formData,
  });
}

export async function deleteDocument(fileName: string): Promise<{
  ok: boolean;
  deleted: DocumentItem;
  vector_chunks_deleted: Record<IndexStrategy, number>;
  processed_chunks_deleted: Record<string, number>;
  corpus_version: number;
  index_version: number;
  pdf_dir: string;
}> {
  return request(`/api/documents/${encodeURIComponent(fileName)}`, {
    method: 'DELETE',
  });
}

export async function deleteWebCorpusItem(url: string): Promise<{
  ok: boolean;
  deleted: DocumentItem;
  vector_chunks_deleted: Record<IndexStrategy, number>;
  processed_chunks_deleted: Record<string, number>;
  corpus_version: number;
  index_version: number;
  web_corpus_dir: string;
}> {
  return request(`/api/documents/web-crawl?url=${encodeURIComponent(url)}`, {
    method: 'DELETE',
  });
}

export async function getRetrievalSettings(): Promise<RetrievalSettings> {
  return request('/api/settings/retrieval');
}

export async function updateRetrievalSettings(settings: RetrievalSettings): Promise<RetrievalSettings> {
  return request('/api/settings/retrieval', {
    method: 'PATCH',
    body: JSON.stringify(settings),
  });
}

export async function ingest(force = false): Promise<{ counts: Record<IndexStrategy, number>; message: string }> {
  return request('/api/ingest', {
    method: 'POST',
    body: JSON.stringify({ force, strategies: ['fixed_size', 'structure_aware'] }),
  });
}

export async function askQuestion(
  question: string,
  session_id?: number | null,
  chunking_strategy?: Strategy,
  top_k?: number,
  run_judge?: boolean,
): Promise<ChatResponse> {
  return request('/api/chat/query', {
    method: 'POST',
    body: JSON.stringify({ question, session_id, chunking_strategy, top_k, run_judge }),
  });
}

export async function getChatHistory(): Promise<{ items: ChatHistorySession[] }> {
  return request('/api/chat/history');
}

export async function getChatSession(sessionId: number): Promise<{ session: ChatHistorySession; items: ChatHistoryItem[] }> {
  return request(`/api/chat/history/${sessionId}`);
}

export async function deleteChatHistoryItem(chat_log_id: number): Promise<{ ok: boolean; deleted: number }> {
  return request(`/api/chat/history/${chat_log_id}`, {
    method: 'DELETE',
  });
}

export async function clearChatHistory(): Promise<{ ok: boolean; deleted: number }> {
  return request('/api/chat/history', {
    method: 'DELETE',
  });
}

export async function sendFeedback(chat_log_id: number, rating: number, comment?: string, feedback_type?: string) {
  return request('/api/chat/feedback', {
    method: 'POST',
    body: JSON.stringify({ chat_log_id, rating, comment, feedback_type }),
  });
}

export async function sendDemoFeedback(payload: DemoFeedbackPayload): Promise<{ ok: boolean; demo_feedback_id: number }> {
  return request('/api/chat/demo-feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function sendHumanAudit(payload: HumanAuditPayload) {
  return request('/api/chat/human-audit', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function trackCitationClick(payload: CitationClickPayload) {
  return request('/api/chat/citation-click', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getAdminMetrics(): Promise<AdminMetrics> {
  return request('/api/chat/admin-metrics');
}

export async function getAdminInsights(includeResolved = false): Promise<AdminInsights> {
  return request(`/api/chat/admin-insights?include_resolved=${includeResolved ? 'true' : 'false'}`);
}

export async function resolveAdminInsight(payload: {
  issue_type: InsightIssueType;
  item_key: string;
  resolved: boolean;
  note?: string;
}): Promise<{ ok: boolean; issue_type: InsightIssueType; item_key: string; resolved: boolean; updated_at: string }> {
  return request('/api/chat/admin-insights/resolve', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getDocumentUrl(fileName: string, pageStart?: number) {
  const pageHash = pageStart ? `#page=${pageStart}` : '';
  return `${API_BASE}/api/documents/file/${encodeURIComponent(fileName)}${pageHash}`;
}

export function getCitationUrl(citation: { file_name: string; page_start?: number | null; source_path?: string | null }) {
  if (citation.source_path?.startsWith('http://') || citation.source_path?.startsWith('https://')) {
    return citation.source_path;
  }
  return getDocumentUrl(citation.file_name, citation.page_start ?? undefined);
}
