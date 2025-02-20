package com.example.service_data.api.external.fmarket.req;

import java.util.List;

import lombok.Data;

@Data
public class FundRequest {
    private Long page = 1L;
    private Long pageSize = 1L;
    private boolean isSmartPortfolio = false;
    private List<String> productTypes;
}
