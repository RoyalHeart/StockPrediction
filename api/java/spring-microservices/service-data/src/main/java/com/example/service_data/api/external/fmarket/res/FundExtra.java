package com.example.service_data.api.external.fmarket.res;

import com.fasterxml.jackson.annotation.JsonAlias;

import lombok.Data;

@Data
public class FundExtra {
    private OrdersSummary ordersSummary;

    @Data
    public class OrdersSummary {
        @JsonAlias("FUND")
        private FUND fund;

        @Data
        public class FUND {
            private long totalCurrentValue;
            private long totalBuyingValue;
            private long totalGain;
            private long totalGainPercent;

        }

    }
}
