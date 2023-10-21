-- insurance_clause.article_section definition

CREATE TABLE `article_section` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `doc_id` varchar(64) DEFAULT NULL,
  `doc_name` varchar(256) DEFAULT NULL,
  `content` longtext,
  `content_type` varchar(255) DEFAULT NULL,
  `length` int DEFAULT NULL,
  `section_id` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=68 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='article section table';


-- insurance_clause.data_embedding definition

CREATE TABLE `data_embedding` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `doc_id` varchar(100) DEFAULT NULL,
  `data_id` varchar(100) DEFAULT NULL,
  `data_vector` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=68 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


-- insurance_clause.section_vector_result definition

CREATE TABLE `section_vector_result` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `doc_id` varchar(64) DEFAULT NULL,
  `data_id` bigint DEFAULT NULL,
  `similarity` decimal(10,8) DEFAULT NULL,
  `query_data` longtext,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='section vector result table';