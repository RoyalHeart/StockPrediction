package com.example.service_data.api.external.binance.service;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.binance.connector.client.SpotClient;
import com.binance.connector.client.impl.SpotClientImpl;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class BinanceService {
    @Value("binance.api_key")
    private static String API_KEY;

    @Value("binance.api_secret")
    private String API_SECRET;

    public Long getBalance() {
        SpotClient client = new SpotClientImpl(API_KEY, API_SECRET);
        Map<String, Object> parameters = new LinkedHashMap<>();
        String result = client.createMarket().exchangeInfo(parameters);
        log.info(">>> " + result);
        return 1L;
    }
}
