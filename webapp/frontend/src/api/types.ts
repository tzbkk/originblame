export interface OverviewResponse {
  records: number;
  sections: number;
  authors: number;
  contributors: number;
  revoked: { authors: number; sections: number; records: number };
  top_authors: Array<{ name: string; sections: number }>;
  author_ranking: string[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
}

export interface AuthorItem {
  id: string;
  name: string;
  email: string;
  sections: number;
  contribution_pct: string;
  revoked: boolean;
}

export interface AuthorDetailResponse {
  author: { id: string; name: string; email: string; revoked: boolean };
  metrics: {
    sections_as_author: number;
    records_as_author: number;
    records_as_contributor: number;
    sections_as_contributor: number;
  };
  author_records: Array<AuthorRecord>;
  author_page: number;
  author_limit: number;
  author_total: number;
  contributor_records: Array<ContributorRecord>;
  contributor_page: number;
  contributor_limit: number;
  contributor_total: number;
}

export interface AuthorRecord {
  title: string;
  heading: string;
  sec_path: string;
  year: string;
  license: string;
  preview: string;
  text: string;
  authors_json: string;
  line_hash?: string;
}

export interface ContributorRecord extends AuthorRecord {
  is_author: boolean;
}

export interface RecordItem {
  title: string;
  heading: string;
  preview: string;
  authors: string;
  year: string;
  status: 'active' | 'revoked';
  sec_path: string;
  line_hash?: string;
}

export interface RecordDetail {
  title: string;
  heading: string;
  year: string;
  license: string;
  revoked: boolean;
  authors: Array<{ id: string; name: string; email: string }>;
  text: string;
  line_hash: string;
  section_hashes: string[];
  sections: Array<{
    section_hash: string;
    path: string;
    license: string;
    year: string;
    revoked: boolean;
    contributors: string[];
  }>;
}

export interface SectionItem {
  section_hash: string;
  path: string;
  title: string;
  heading: string;
  authors: string;
  license: string;
  year: string;
  revoked: boolean;
  record_count: number;
}

export interface ErasureImpact {
  target_name: string;
  is_already_revoked: boolean;
  affected_sections: number;
  affected_records: number;
  affected_contrib_sections: number;
  affected_contrib_records: number;
  total_sections: number;
  total_records: number;
  revoke_desc: string;
  comparison: {
    file_level: number;
    contributor_level: number;
    record_level: number;
    factor: number;
  } | null;
}

export interface RevokedData {
  revoked_authors: Array<{
    id: string;
    name: string;
    email: string;
    affected_sections: number;
  }>;
  revoked_sections: Array<{
    section_hash: string;
    path: string;
    title: string;
    heading: string;
    authors: string;
    record_count: number;
  }>;
  cascade_count: number;
}

export interface AuditEntry {
  ts: string;
  op: string;
  detail: string | Record<string, unknown>;
  cmd?: string;
}

export interface MessageResponse {
  message: string;
}

export interface DatasetsResponse {
  datasets: string[];
  default: string;
}
