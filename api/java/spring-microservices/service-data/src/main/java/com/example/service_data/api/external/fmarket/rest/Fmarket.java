/*******************************************************************************
 * Class        :Fmarket
 * Created date :2025/02/19
 * Lasted date  :2025/02/19
 * Author       :TamTH1
 * Change log   :2025/02/19 01-00 TamTH1 create a new
******************************************************************************/
package com.example.service_data.api.external.fmarket.rest;

import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClient.RequestHeadersSpec;

import com.example.service_data.api.external.fmarket.req.FundRequest;
import com.example.service_data.api.external.fmarket.req.LoginRequest;
import com.example.service_data.api.external.fmarket.res.FundResponse;
import com.example.service_data.api.external.fmarket.res.LoginResponse;

/**
 * Fmarket
 * 
 * @version 01-00
 * @since 01-00
 * @author TamTH1
 */
@Service
public class Fmarket {
    @Value("${fmarket.email}")
    private String EMAIL;
    @Value("${fmarket.password}")
    private String PASSWORD;
    @Autowired
    RestClient restClient;

    private final String LOGIN_URL = "/auth/login";
    private final String FUND_URL = "/v2.1/investors/assets/fund";
    private final String FMARKET_HOST = "https://api.fmarket.vn";

    public LoginResponse login(LoginRequest loginRequest) {
        RestClient restClient = RestClient.builder().baseUrl(FMARKET_HOST).build();
        RequestHeadersSpec<?> request = restClient.post()
                .uri(LOGIN_URL)
                .body(loginRequest)
                .headers(headers -> {
                    headers.add("Accept", "application/json");
                    headers.add("User-Agent",
                            "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36");
                }).accept(MediaType.APPLICATION_JSON);
        LoginResponse response = request.retrieve().toEntity(LoginResponse.class).getBody();
        return response;
    }

    public FundResponse getFund() {
        FundRequest fundRequest = new FundRequest();
        List<String> productTypes = new ArrayList<>();
        productTypes.add("NEW_FUND");
        productTypes.add("TRADING_FUND");
        fundRequest.setProductTypes(productTypes);
        RestClient restClient = RestClient.builder().baseUrl(FMARKET_HOST).build();
        LoginRequest loginRequest = LoginRequest.builder().email(EMAIL).password(PASSWORD).build();
        LoginResponse loginResponse = login(loginRequest);
        String accessToken = "Bearer " + loginResponse.getData().getAccessToken();
        RequestHeadersSpec<?> request = restClient.post()
                .uri(FUND_URL)
                .body(fundRequest)
                .headers(headers -> {
                    headers.add("Accept", "application/json");
                    headers.add("User-Agent",
                            "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36");
                    headers.add("Authorization",
                            accessToken);
                }).accept(MediaType.APPLICATION_JSON);
        FundResponse response = request.retrieve().toEntity(FundResponse.class).getBody();
        return response;
    }

    public Long getBalance() {
        FundResponse fundResponse = getFund();
        return fundResponse.getExtra().getOrdersSummary().getFund().getTotalCurrentValue();
    }
}
