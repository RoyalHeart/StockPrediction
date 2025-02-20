package com.example.service_data.api.external.fmarket.req;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class LoginRequest {
    private String email;
    private String password;
    @JsonProperty
    private String referralCode;
}
