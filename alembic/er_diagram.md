# ER diagram
```mermaid
erDiagram
    alembic_version {
        character_varying version_num PK
    }

    analyze_request_metrics {
        timestamp_with_time_zone analysis_completed_at
        endpointtype endpoint
        integer id PK
        integer merge_request_job_id FK
        timestamp_with_time_zone request_received_at
        integer response_length
        timestamp_with_time_zone response_sent_at
    }

    annotated_builds {
        bigint id PK
        character_varying problem
        character_varying solution
        character_varying source_path UK
    }

    annotated_snippets {
        character_varying annotation
        bigint id PK
        character_varying source_artifact_name
        bigint source_build_id FK
        character_varying text
        vector text_embedding
    }

    annotation_updates {
        date archive_date
        integer file_count
        bigint id PK
    }

    comments {
        character_varying comment_id UK
        timestamp_with_time_zone created_at
        forge forge UK
        bigint id PK
        bigint merge_request_job_id FK
    }

    gitlab_merge_request_jobs {
        forge forge UK
        bigint id PK
        bigint job_id UK
        bigint mr_iid UK
        bigint project_id UK
    }

    task_analysis {
        integer attempt_count
        character_varying external_task_id
        integer id PK
        timestamp_with_time_zone request_received_at
        bytea response
        integer response_metrics_id FK
        timestamp_with_time_zone response_returned_at
        analysisstate state
        uuid task_id
        json task_metadata
        tasktype task_type
    }

    analyze_request_metrics }o--|| gitlab_merge_request_jobs : "merge_request_job_id"
    task_analysis }o--|| analyze_request_metrics : "response_metrics_id"
    annotated_snippets }o--|| annotated_builds : "source_build_id"
    comments }o--|| gitlab_merge_request_jobs : "merge_request_job_id"
```
