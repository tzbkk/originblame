import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from './client';
import type {
  OverviewResponse,
  PaginatedResponse,
  AuthorItem,
  AuthorDetailResponse,
  RecordItem,
  RecordDetail,
  SectionItem,
  ErasureImpact,
  RevokedData,
  AuditEntry,
  MessageResponse,
  DatasetsResponse,
} from './types';

export const queryKeys = {
  datasets: ['datasets'] as const,
  overview: (ds: string) => ['overview', ds] as const,
  authors: (ds: string, params: Record<string, unknown>) =>
    ['authors', ds, params] as const,
  authorDetail: (ds: string, id: string, ap = 1, cp = 1, limit = 20) =>
    ['authorDetail', ds, id, ap, cp, limit] as const,
  records: (ds: string, params: Record<string, unknown>) =>
    ['records', ds, params] as const,
  recordDetail: (ds: string, hash: string, secPath: string = '') =>
    ['recordDetail', ds, hash, secPath] as const,
  sections: (ds: string, params: Record<string, unknown>) =>
    ['sections', ds, params] as const,
  erasureImpact: (ds: string, type: string, target: string) =>
    ['erasureImpact', ds, type, target] as const,
  revoked: (ds: string) => ['revoked', ds] as const,
  auditLog: (ds: string, op?: string) =>
    ['auditLog', ds, op] as const,
};

export function useDatasets() {
  return useQuery({
    queryKey: queryKeys.datasets,
    queryFn: () => apiFetch<DatasetsResponse>('/datasets'),
  });
}

export function useOverview(dataset: string) {
  return useQuery({
    queryKey: queryKeys.overview(dataset),
    queryFn: () =>
      apiFetch<OverviewResponse>(`/${dataset}/overview`),
    enabled: !!dataset,
  });
}

export function useAuthors(
  dataset: string,
  params: { search?: string; page?: number; limit?: number },
) {
  const sp = new URLSearchParams();
  if (params.search) sp.set('search', params.search);
  if (params.page) sp.set('page', String(params.page));
  if (params.limit) sp.set('limit', String(params.limit));
  return useQuery({
    queryKey: queryKeys.authors(dataset, params),
    queryFn: () =>
      apiFetch<PaginatedResponse<AuthorItem>>(
        `/${dataset}/authors?${sp.toString()}`,
      ),
    enabled: !!dataset,
  });
}

export function useAuthorDetail(
  dataset: string,
  authorId: string,
  authorPage = 1,
  contributorPage = 1,
  limit = 20,
) {
  return useQuery({
    queryKey: queryKeys.authorDetail(dataset, authorId, authorPage, contributorPage, limit),
    queryFn: () =>
      apiFetch<AuthorDetailResponse>(
        `/${dataset}/authors/${authorId}?author_page=${authorPage}&author_limit=${limit}&contributor_page=${contributorPage}&contributor_limit=${limit}`,
      ),
    enabled: !!dataset && !!authorId,
  });
}

export function useRecords(
  dataset: string,
  params: {
    search?: string;
    author?: string;
    status?: string;
    year?: string;
    page?: number;
    limit?: number;
  },
) {
  const sp = new URLSearchParams();
  if (params.search) sp.set('search', params.search);
  if (params.author) sp.set('author', params.author);
  if (params.status) sp.set('status', params.status);
  if (params.year) sp.set('year', params.year);
  if (params.page) sp.set('page', String(params.page));
  if (params.limit) sp.set('limit', String(params.limit));
  return useQuery({
    queryKey: queryKeys.records(dataset, params),
    queryFn: () =>
      apiFetch<PaginatedResponse<RecordItem>>(
        `/${dataset}/records?${sp.toString()}`,
      ),
    enabled: !!dataset,
  });
}

export function useRecordDetail(dataset: string, hash: string = "", secPath: string = "") {
  return useQuery({
    queryKey: queryKeys.recordDetail(dataset, hash),
    queryFn: () =>
      apiFetch<RecordDetail>(
        `/${dataset}/records/detail?hash=${encodeURIComponent(hash)}&sec_path=${encodeURIComponent(secPath)}`,
      ),
    enabled: !!dataset && (!!hash || !!secPath),
  });
}

export function useSections(
  dataset: string,
  params: { page?: number; limit?: number; revoked_only?: boolean },
) {
  const sp = new URLSearchParams();
  if (params.page) sp.set('page', String(params.page));
  if (params.limit) sp.set('limit', String(params.limit));
  if (params.revoked_only) sp.set('revoked_only', 'true');
  return useQuery({
    queryKey: queryKeys.sections(dataset, params),
    queryFn: () =>
      apiFetch<PaginatedResponse<SectionItem>>(
        `/${dataset}/sections?${sp.toString()}`,
      ),
    enabled: !!dataset,
  });
}

export function useErasureImpact(
  dataset: string,
  revokeType: string,
  target: string,
) {
  return useQuery({
    queryKey: queryKeys.erasureImpact(dataset, revokeType, target),
    queryFn: () =>
      apiFetch<ErasureImpact>(
        `/${dataset}/erasure/impact?revoke_type=${revokeType}&target=${encodeURIComponent(target)}`,
      ),
    enabled: !!dataset && !!revokeType && !!target,
  });
}

export function useRevoked(dataset: string) {
  return useQuery({
    queryKey: queryKeys.revoked(dataset),
    queryFn: () => apiFetch<RevokedData>(`/${dataset}/revoked`),
    enabled: !!dataset,
  });
}

export function useAuditLog(dataset: string, op?: string) {
  const suffix = op ? `?op=${op}` : '';
  return useQuery({
    queryKey: queryKeys.auditLog(dataset, op),
    queryFn: () =>
      apiFetch<{ entries: AuditEntry[]; total: number }>(
        `/${dataset}/audit-log${suffix}`,
      ),
    enabled: !!dataset,
  });
}

export function useRevokeAuthor(dataset: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (email: string) =>
      apiFetch<MessageResponse>(`/${dataset}/revoke/author`, {
        method: 'POST',
        body: JSON.stringify({ email }),
      }),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useRevokeSection(dataset: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sectionHash: string) =>
      apiFetch<MessageResponse>(`/${dataset}/revoke/section`, {
        method: 'POST',
        body: JSON.stringify({ section_hash: sectionHash }),
      }),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useRevokeRecord(dataset: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (path: string) =>
      apiFetch<MessageResponse>(`/${dataset}/revoke/record`, {
        method: 'POST',
        body: JSON.stringify({ path }),
      }),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useRestoreAuthor(dataset: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (authorId: string) =>
      apiFetch<MessageResponse>(`/${dataset}/restore/author`, {
        method: 'POST',
        body: JSON.stringify({ author_id: authorId }),
      }),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useRestoreSection(dataset: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sectionHash: string) =>
      apiFetch<MessageResponse>(`/${dataset}/restore/section`, {
        method: 'POST',
        body: JSON.stringify({ section_hash: sectionHash }),
      }),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useResetDemo(dataset: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<MessageResponse>(`/${dataset}/reset`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries(),
  });
}
